from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import os
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
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

# Emotion-to-genre mappings (updated to match AffectNet labels)
emotion_genre_map = {
    'anger':     {'movie': 'Comedy',     'music': 'chill'},
    'disgust':   {'movie': 'Adventure',  'music': 'acoustic'},
    'fear':      {'movie': 'Fantasy',    'music': 'classical'},
    'happiness': {'movie': 'Musical',    'music': 'pop'},
    'neutral':   {'movie': 'Drama',      'music': 'indie'},
    'sadness':   {'movie': 'Romance',    'music': 'uplifting'},
    'surprise':  {'movie': 'Thriller',   'music': 'edm'},
    'contempt':  {'movie': 'Dark Comedy','music': 'alternative'}
}

# Emotion labels (matching AffectNet 8 categories)
emotion_labels = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

# Proper AffectNet Model (matching the training code)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super(EmotionCNN, self).__init__()
        
        # Use ResNet18 as backbone (same as training code)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer for 8 emotions
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Load model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN(num_classes=len(emotion_labels), pretrained=False)

# Try to load the trained model weights
model_path = "best_emotion_model.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("Loaded trained emotion model successfully")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.info("Using untrained model - please train the model first!")
else:
    logging.warning(f"Model file {model_path} not found. Using untrained model.")
    logging.info("Please run the training script first to generate the model weights.")

model.to(device)
model.eval()

# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection

# Image preprocessing (matching training transforms)
def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Emotion detection using PyTorch
def detect_emotion(image_np):
    try:
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
            # Convert BGR to RGB if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_rgb = image_np
            else:
                image_rgb = np.array(Image.fromarray(image_np).convert('RGB'))
                
            results = face_detection.process(image_rgb)

            if not results.detections:
                logging.warning("No face detected.")
                return "neutral", 0.0

            # Get the largest face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image_rgb.shape
            
            # Calculate face bounding box
            x = max(0, int(bbox.xmin * iw))
            y = max(0, int(bbox.ymin * ih))
            w = min(int(bbox.width * iw), iw - x)
            h = min(int(bbox.height * ih), ih - y)
            
            # Extract face region
            face = image_rgb[y:y+h, x:x+w]
            
            if face.size == 0:
                logging.warning("Empty face region extracted.")
                return "neutral", 0.0
            
            # Convert to PIL Image and preprocess
            face_pil = Image.fromarray(face)
            transform = get_inference_transform()
            input_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Predict emotion
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, emotion_idx = torch.max(probs, dim=1)
                
                emotion = emotion_labels[emotion_idx.item()]
                confidence_score = confidence.item()
                
                logging.info(f"Detected emotion: {emotion} (confidence: {confidence_score:.3f})")
                return emotion, confidence_score
                
    except Exception as e:
        logging.error(f"Emotion detection error: {e}")
        return "neutral", 0.0

# Recommendation logic
def get_recommendations(emotion):
    # Map emotion to genres (handle both old and new naming)
    emotion_key = emotion.lower()
    if emotion_key == 'happy':
        emotion_key = 'happiness'
    elif emotion_key == 'sad':
        emotion_key = 'sadness'
    elif emotion_key == 'angry':
        emotion_key = 'anger'
    
    genres = emotion_genre_map.get(emotion_key, {'movie': 'Drama', 'music': 'pop'})
    movie_genre, music_genre = genres['movie'], genres['music']

    # OMDb Recommendations
    movies = []
    try:
        movie_url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={movie_genre}&type=movie&y=2020"
        movie_response = requests.get(movie_url, timeout=10)
        
        if movie_response.status_code == 200:
            movie_data = movie_response.json()
            if movie_data.get("Response") == "True":
                movies = [
                    {
                        "title": movie["Title"],
                        "year": movie.get("Year", "N/A"),
                        "poster": movie.get("Poster", "N/A")
                    }
                    for movie in movie_data.get("Search", [])[:5]
                ]
            else:
                logging.warning(f"OMDb API returned error: {movie_data.get('Error', 'Unknown error')}")
        else:
            logging.error(f"OMDb API request failed with status: {movie_response.status_code}")
            
    except Exception as e:
        logging.error(f"OMDb Error: {e}")

    # Spotify Recommendations
    songs = []
    try:
        # Try genre-based search first
        results = sp.search(q=f"genre:{music_genre}", type="track", limit=10)
        
        # If no results, try mood-based search
        if not results["tracks"]["items"]:
            results = sp.search(q=f"{music_genre} mood", type="track", limit=10)
        
        for track in results["tracks"]["items"][:5]:
            song_info = {
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "preview": track.get("preview_url"),
                "spotify_url": track["external_urls"]["spotify"],
                "album": track["album"]["name"]
            }
            songs.append(song_info)
            
    except Exception as e:
        logging.error(f"Spotify Error: {e}")

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
        
        # Load and convert image
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_np = np.array(image)

        # Detect emotion
        emotion, confidence = detect_emotion(image_np)
        
        # Get recommendations
        recs = get_recommendations(emotion)

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "recommendations": recs,
            "status": "success"
        })
        
    except Exception as e:
        logging.error(f"Unexpected error in /detect_emotion: {e}")
        return jsonify({'error': 'Failed to process the image', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_loaded = os.path.exists("best_emotion_model.pth")
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "emotion_labels": emotion_labels
    })

@app.route('/test_model', methods=['GET'])
def test_model():
    """Test endpoint to check model setup"""
    try:
        # Create a dummy input to test model
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            probs = F.softmax(output, dim=1)
        
        return jsonify({
            "status": "model_working",
            "output_shape": list(output.shape),
            "sample_probabilities": probs[0].cpu().numpy().tolist()
        })
    except Exception as e:
        return jsonify({
            "status": "model_error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
