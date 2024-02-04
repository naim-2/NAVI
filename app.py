from flask import Flask, render_template, Response
import requests
import cv2
from gtts import gTTS
import pygame
import time
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Azure Cognitive Services API Key and Endpoint
api_key = "b7bb5fe0d6da439181ed81904b28e959"
endpoint = "https://nsalim-vision-api.cognitiveservices.azure.com/"

# Video feed (replace 0 with the camera index if using a webcam)
video_feed = 0

# Initialize pygame mixer
pygame.mixer.init()

def analyze_frame(frame):
    # Convert the frame to bytes
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    # Azure Cognitive Services API endpoint for image analysis
    analyze_url = f"{endpoint}/vision/v3.1/analyze"

    # Parameters for the analysis (change as needed)
    params = {
        'visualFeatures': 'Description',
        'language': 'en'
    }

    # Headers with the API key
    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': api_key,
    }

    # Make the API request
    response = requests.post(analyze_url, params=params, headers=headers, data=img_bytes)

    # Get the analysis results
    results = response.json()

    return results

def generate_frames():
    cap = cv2.VideoCapture(video_feed)
    audio_timer = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = analyze_frame(frame)

            # Display the analysis results
            if 'description' in results:
                description = results['description']['captions'][0]['text']
                cv2.putText(frame, f"Description: {description}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Check if 3 seconds have passed since the last audio generation
                if time.time() - audio_timer >= 3:
                    # Convert description to speech
                    tts = gTTS(text=description, lang='en', slow=False)
                    current_datetime = datetime.now()
                    current_time = current_datetime.timestamp()
                    tts.save('static/output'+str(current_time)+'.mp3')

                    # Play the generated audio using pygame mixer
                    pygame.mixer.music.load('static/output'+str(current_time)+'.mp3')
                    pygame.mixer.music.play()

                    # Reset the timer
                    audio_timer = time.time()

            # Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
