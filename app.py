import streamlit as st
import requests
import cv2
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MoodSync - AI-Powered Mood Detection", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .emotion-display {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 1rem 0;
    }
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 25px;
        padding: 2px;
        margin: 10px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 20px;
        border-radius: 25px;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000"

# Header
st.markdown('<h1 class="main-header">🎭 MoodSync</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Emotion Detection & Entertainment Recommendations</p>', unsafe_allow_html=True)

# Check backend health
def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else {}
    except Exception as e:
        return False, {"error": str(e)}

# Backend status indicator
with st.expander("🔧 System Status", expanded=False):
    health_status, health_data = check_backend_health()
    
    if health_status:
        st.success("✅ Backend is running")
        st.json(health_data)
    else:
        st.error("❌ Backend is not accessible")
        st.warning("Please make sure the Flask backend is running on http://127.0.0.1:5000")

# Main content
st.markdown("---")

# Columns for webcam and mood display
col1, col2 = st.columns([1.2, 1])
emotion_response = {}

# Webcam section
with col1:
    st.markdown("### 📸 Emotion Detection")
    
    # File upload option
    upload_option = st.radio("Choose input method:", ["📷 Use Webcam", "📁 Upload Image"])
    
    if upload_option == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Emotion"):
                with st.spinner("Analyzing emotion..."):
                    try:
                        # Send image to backend
                        files = {"image": uploaded_file.getvalue()}
                        response = requests.post(f"{BACKEND_URL}/detect_emotion", 
                                               files={"image": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")})
                        
                        if response.status_code == 200:
                            emotion_response = response.json()
                            # Store in history
                            st.session_state.emotion_history.append({
                                'timestamp': datetime.now(),
                                'emotion': emotion_response.get('emotion', 'neutral'),
                                'confidence': emotion_response.get('confidence', 0.0)
                            })
                            st.success("✅ Emotion detected successfully!")
                        else:
                            st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")
    
    else:  # Webcam option
        capture_btn = st.button("📸 Capture & Analyze Emotion")
        
        # Webcam feed
        FRAME_WINDOW = st.empty()
        
        try:
            camera = cv2.VideoCapture(0)
            ret, frame = camera.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, caption="Live Webcam Feed")
                
                if capture_btn:
                    with st.spinner("Capturing and analyzing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                            cv2.imwrite(temp_file.name, frame)
                            
                            # Send image to backend
                            with open(temp_file.name, "rb") as f:
                                files = {"image": f}
                                response = requests.post(f"{BACKEND_URL}/detect_emotion", files=files)
                            
                            # Clean up temp file
                            os.unlink(temp_file.name)
                            
                            if response.status_code == 200:
                                emotion_response = response.json()
                                # Store in history
                                st.session_state.emotion_history.append({
                                    'timestamp': datetime.now(),
                                    'emotion': emotion_response.get('emotion', 'neutral'),
                                    'confidence': emotion_response.get('confidence', 0.0)
                                })
                                st.success("✅ Emotion detected successfully!")
                            else:
                                st.error(f"❌ Error: {response.json().get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ Could not access webcam. Please check your camera permissions.")
            
            camera.release()
            
        except Exception as e:
            st.error(f"❌ Webcam error: {str(e)}")

# Mood display section
with col2:
    st.markdown("### 🎯 Detected Emotion")
    
    if emotion_response:
        emotion = emotion_response.get("emotion", "neutral")
        confidence = emotion_response.get("confidence", 0.0)
        
        # Updated mood mapping for AffectNet emotions
        mood_map = {
            "happiness": ("😄", "Feeling great! Let's keep the positive vibes flowing."),
            "sadness": ("😢", "It's okay to feel down. Here's something uplifting."),
            "anger": ("😠", "Take a deep breath. Let's cool down with something relaxing."),
            "surprise": ("😲", "Whoa! That's unexpected. Let's match that energy."),
            "fear": ("😰", "Feeling anxious? We've got some calming content."),
            "neutral": ("😐", "Balanced mood. Let's discover something interesting!"),
            "disgust": ("🤢", "Let's brighten your mood with something pleasant."),
            "contempt": ("😒", "Not impressed? Let's find something that sparks joy.")
        }
        
        icon, message = mood_map.get(emotion.lower(), ("🙂", "Let's find something perfect for you."))
        
        # Emotion display with styling
        st.markdown(f"""
        <div class="emotion-display">
            <h2>{icon}</h2>
            <h3 style="color: #667eea; margin: 0;">{emotion.title()}</h3>
            <p style="margin: 0.5rem 0; color: #666;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        st.markdown("**Confidence Level:**")
        confidence_percent = int(confidence * 100)
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percent}%;">
                {confidence_percent}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("👆 Capture an image or upload a photo to detect your emotion!")

# Recommendations section
st.markdown("---")
st.markdown("## 🎯 Personalized Recommendations")

if emotion_response and emotion_response.get("recommendations"):
    recs = emotion_response["recommendations"]
    movies = recs.get("movies", [])
    songs = recs.get("songs", [])
    
    # Create tabs for different recommendation types
    movie_tab, music_tab = st.tabs(["🎬 Movies", "🎵 Music"])
    
    # Movies tab
    with movie_tab:
        if movies:
            st.markdown(f"**Based on your {emotion_response.get('emotion', 'current')} mood:**")
            
            for movie in movies:
                if isinstance(movie, dict):
                    title = movie.get("title", "Unknown Movie")
                    year = movie.get("year", "")
                    poster = movie.get("poster", "")
                    
                    col_poster, col_info = st.columns([1, 3])
                    
                    with col_poster:
                        if poster and poster != "N/A":
                            st.image(poster, width=100)
                    
                    with col_info:
                        st.markdown(f"**🎬 {title}**")
                        if year:
                            st.markdown(f"*Released: {year}*")
                        st.markdown("---")
                else:
                    # Handle simple string format
                    st.markdown(f"🎬 **{movie}**")
        else:
            st.info("🔍 No movie recommendations available right now.")
    
    # Music tab
    with music_tab:
        if songs:
            st.markdown(f"**Curated playlist for your {emotion_response.get('emotion', 'current')} mood:**")
            
            for i, song in enumerate(songs, 1):
                title = song.get("title", "Unknown Song")
                artist = song.get("artist", "Unknown Artist")
                album = song.get("album", "")
                preview = song.get("preview")
                spotify_url = song.get("spotify_url")
                
                # Song card
                with st.container():
                    st.markdown(f"**{i}. 🎵 {title}**")
                    st.markdown(f"*by {artist}*")
                    
                    if album:
                        st.markdown(f"*Album: {album}*")
                    
                    col_audio, col_link = st.columns([2, 1])
                    
                    with col_audio:
                        if preview:
                            st.audio(preview, format="audio/mp3")
                        else:
                            st.info("No preview available")
                    
                    with col_link:
                        if spotify_url:
                            st.markdown(f"[🎧 Open in Spotify]({spotify_url})")
                    
                    st.markdown("---")
        else:
            st.info("🎼 No music recommendations available right now.")
    
else:
    st.warning("🎯 Recommendations will appear after emotion detection.")

# Feedback section
st.markdown("---")
st.markdown("## 💬 Feedback")

if emotion_response:
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        if st.button("👍 Accurate Detection"):
            st.session_state.feedback_history.append({
                'timestamp': datetime.now(),
                'emotion': emotion_response.get('emotion'),
                'feedback': 'positive',
                'accurate': True
            })
            st.success("Thank you for your feedback!")
    
    with feedback_col2:
        if st.button("👎 Inaccurate Detection"):
            st.session_state.feedback_history.append({
                'timestamp': datetime.now(),
                'emotion': emotion_response.get('emotion'),
                'feedback': 'negative',
                'accurate': False
            })
            st.info("Thanks! This helps us improve our model.")
    
    # Additional feedback
    feedback_text = st.text_area("💭 Additional thoughts or suggestions (optional):")
    
    if st.button("📤 Submit Detailed Feedback"):
        feedback_data = {
            "emotion": emotion_response.get("emotion", "unknown"),
            "confidence": emotion_response.get("confidence", 0.0),
            "recommendations": emotion_response.get("recommendations", {}),
            "comment": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Try to send to backend (you'll need to implement this endpoint)
            response = requests.post(f"{BACKEND_URL}/submit_feedback", json=feedback_data, timeout=5)
            if response.status_code == 200:
                st.success("✅ Feedback submitted successfully!")
            else:
                st.warning("⚠️ Feedback saved locally (backend not available)")
        except:
            # Store locally if backend is not available
            st.session_state.feedback_history.append(feedback_data)
            st.warning("⚠️ Feedback saved locally (backend not available)")

# Analytics section
st.markdown("---")
st.markdown("## 📊 Your Emotion Analytics")

if st.session_state.emotion_history:
    # Create emotion frequency chart
    emotion_counts = {}
    for entry in st.session_state.emotion_history:
        emotion = entry['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    if emotion_counts:
        # Create a more attractive chart
        fig, ax = plt.subplots(figsize=(10, 6))
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        
        # Use seaborn for better styling
        colors = sns.color_palette("husl", len(emotions))
        bars = ax.bar(emotions, counts, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_title("Your Emotion History", fontsize=16, fontweight='bold')
        ax.set_xlabel("Emotions", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Show recent detections
        st.markdown("### 🕒 Recent Detections")
        recent_emotions = st.session_state.emotion_history[-5:]  # Last 5
        
        for entry in reversed(recent_emotions):
            timestamp = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            emotion = entry['emotion']
            confidence = entry['confidence']
            st.markdown(f"• **{emotion.title()}** ({confidence:.1%} confidence) - *{timestamp}*")
else:
    st.info("📈 Your emotion analytics will appear here after a few detections!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎭 MoodSync - Powered by AI Emotion Recognition</p>
    <p><small>Your privacy is important. Images are processed locally and not stored.</small></p>
</div>
""", unsafe_allow_html=True)
