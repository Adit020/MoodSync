from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import os
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
from tensorflow.keras.models import model_from_json
import mediapipe as mp

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Emotion-to-genre mappings
emotion_genre_map = {
    'angry':     {'movie': 'Comedy',     'music': 'chill'},
    'disgust':   {'movie': 'Adventure',  'music': 'acoustic'},
    'fear':      {'movie': 'Fantasy',    'music': 'classical'},
    'happy':     {'movie': 'Musical',    'music': 'pop'},
    'neutral':   {'movie': 'Drama',      'music': 'indie'},
    'sad':       {'movie': 'Romance',    'music': 'uplifting'},
    'surprise':  {'movie': 'Thriller',   'music': 'edm'},
}

# Load emotion detection model
model = model_from_json(open("model_architecture.json", "r").read())
model.load_weights("model_weights.weights.h5")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection

# Emotion detection using MediaPipe
def detect_emotion(image_np):
    try:
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
            results = face_detection.process(image_np)

            if not results.detections:
                logging.warning("No face detected.")
                return "neutral"

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image_np.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            face = image_np[y:y + h, x:x + w]
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(gray, (48, 48))
            face_resized = face_resized.reshape(1, 48, 48, 1).astype('float32') / 255.0

            prediction = model.predict(face_resized, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            logging.info(f"Detected emotion: {emotion}")
            return emotion
    except Exception as e:
        logging.error(f"Emotion detection error: {e}")
        return "neutral"

# Recommendation system logic with Spotify previews
def get_recommendations(emotion):
    genres = emotion_genre_map.get(emotion.lower(), {'movie': 'Drama', 'music': 'pop'})
    movie_genre, music_genre = genres['movie'], genres['music']

    # OMDb Recommendations
    try:
        movie_url = f"http://www.omdbapi.com/?apikey=4c0baff9&s={movie_genre}&type=movie"
        movie_response = requests.get(movie_url)
        movie_data = movie_response.json()
        movies = [movie["Title"] for movie in movie_data.get("Search", [])[:5]]
    except Exception as e:
        logging.error(f"OMDb Error: {e}")
        movies = []

    # Spotify Recommendations with audio preview
    songs = []
    try:
        results = sp.search(q=f"genre:{music_genre}", type="track", limit=5)
        for track in results["tracks"]["items"]:
            song_info = {
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "preview": track["preview_url"]  # May be None
            }
            songs.append(song_info)
    except Exception as e:
        logging.error(f"Spotify Error: {e}")
        songs = []

    return {
        "movies": movies,
        "songs": songs
    }

@app.route('/detect_emotion', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        img_bytes = image_file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_np = np.array(image)

        emotion = detect_emotion(image_np)
        recs = get_recommendations(emotion)

        return jsonify({
            "emotion": emotion,
            "recommendations": recs
        })
    except Exception as e:
        logging.error(f"Unexpected error in /detect_emotion: {e}")
        return jsonify({'error': 'Failed to process the image'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

