from flask import Flask, render_template, Response
import requests
import cv2
from gtts import gTTS
import pygame
import time
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContentSettings
import io
from io import BytesIO

app = Flask(__name__)

# Azure Cognitive Services API Key and Endpoint
api_key = "9e0adf5fbb5c4e44808a70ecf17daf07"
endpoint = "https://nsalim-api-vision.cognitiveservices.azure.com/"

account_name = "audi0st0rage"
account_key = "rGvae6a1FrcXaEzWBsGOg9I5DOqHXYz7Sng/s6sbBFqZNbSK6nNep0XQAIGKukrrlB+v1GkduAOP+AStSrJhoA=="
container_name = "audio-files"

# Create a BlobServiceClient
blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
container_client = blob_service_client.get_container_client(container_name)

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
    message_timer = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = analyze_frame(frame)

            # Display the analysis results
            if 'description' in results:
                description = results['description']['captions'][0]['text']

                # Check if 2 seconds have passed since the last audio generation
                if time.time() - message_timer >= 2:
                    description = results['description']['captions'][0]['text']
                    cv2.putText(frame, f"Description: {description}", 
                                (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 255, 255), 2, 
                                cv2.LINE_AA)
                    
                    # Reset the timer
                    message_timer = time.time()

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
