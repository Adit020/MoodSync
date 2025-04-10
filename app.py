import streamlit as st
import requests
import cv2
import tempfile
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="MoodSync", layout="centered")

st.markdown("<h1 style='text-align: center;'>MoodSync</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Emotion-Based Entertainment Recommendations</h4>", unsafe_allow_html=True)
st.markdown("---")

# Columns for webcam and mood
col1, col2 = st.columns([1.2, 1])
emotion_response = {}

# Webcam UI
with col1:
    st.markdown("### 📸 Webcam Feed")
    capture_btn = st.button("🎥 Capture & Analyze")

    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    if capture_btn and ret:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, frame)
            image_path = temp_file.name

            # POST image to backend
            backend_url = "http://127.0.0.1:5000/detect_emotion"
            with open(image_path, "rb") as f:
                files = {"image": f}
                response = requests.post(backend_url, files=files)

            if response.status_code == 200:
                emotion_response = response.json()
            else:
                st.error("Failed to fetch emotion from backend.")
    camera.release()

# Mood Display
with col2:
    st.markdown("### 🔍 Current Mood")

    if emotion_response:
        emotion = emotion_response.get("emotion", "neutral").capitalize()
        mood_map = {
            "Happy": ("😄", "Feeling great! Let’s keep the vibes up."),
            "Sad": ("😢", "It's okay to feel down. Here's something uplifting."),
            "Angry": ("😠", "Breathe. Let's cool it down with something chill."),
            "Surprise": ("😲", "Whoa! Let’s match that energy."),
            "Fear": ("⚡", "Feeling anxious? We’ve got calm content."),
            "Neutral": ("😐", "Let’s discover something new!"),
            "Disgust": ("🤢", "Let's brighten the mood."),
        }
        icon, message = mood_map.get(emotion, ("🙂", "Let's find something for you."))

        st.markdown(f"<h2 style='text-align: center;'>{icon}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: purple;'>{emotion}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: gray;'>{message}</p>", unsafe_allow_html=True)
    else:
        st.info("Capture an image to detect emotion.")

st.markdown("---")
st.markdown("## 🎯 Recommendations")

if emotion_response:
    tabs = st.tabs(["🎬 Movies", "🎵 Songs"])

    # Handle format with `recommendations` key
    recs = emotion_response.get("recommendations", {})
    movies = recs.get("movies", [])
    songs = recs.get("songs", [])

    # Movie Tab
    with tabs[0]:
        if movies:
            for movie in movies:
                movie_title = movie
                movie_url = f"http://www.omdbapi.com/?apikey=4c0baff9&s={movie_title}"
                st.markdown(f"• 🎬 [**{movie_title}**]({movie_url})")
        else:
            st.info("No movie recommendations found.")

    # Songs Tab with Audio
    with tabs[1]:
        if songs:
            for song in songs:
                title = song.get("title", "Unknown Song")
                artist = song.get("artist", "Unknown Artist")
                preview = song.get("preview")

                st.markdown(f"**🎵 {title}** by *{artist}*")
                if preview:
                    st.audio(preview)
                st.markdown("---")
        else:
            st.info("No song recommendations found.")
else:
    st.warning("Recommendations will appear after emotion detection.")

# Feedback Section
st.markdown("---")
st.markdown("## 🙋 Was this helpful?")
col1, col2 = st.columns([0.1, 0.1])
with col1:
    if st.button("👍 Yes"):
        st.session_state["liked_feedback"] = True
with col2:
    if st.button("👎 No"):
        st.session_state["liked_feedback"] = False


feedback = st.text_area("Any additional thoughts? (optional)")
if st.button("📤 Send Feedback"):
    feedback_data = {
        "emotion": emotion_response.get("emotion", "unknown"),
        "recommendations": emotion_response.get("recommendations", {}),
        "liked": st.session_state.get("liked_feedback", None),
        "comment": feedback
    }

    try:
        feedback_response = requests.post("http://127.0.0.1:5000/submit_feedback", json=feedback_data)
        if feedback_response.status_code == 200:
            st.success("✅ Feedback submitted for learning. Thank you!")
        else:
            st.error("⚠️ Failed to submit feedback.")
    except:
        st.error("⚠️ Backend not reachable.")

# You could fetch this from Flask too!
@st.cache_data
def get_mock_mood_history():
    return {
        "Happy": 5,
        "Sad": 2,
        "Angry": 1,
        "Surprise": 3,
        "Neutral": 4,
        "Fear": 1
    }

st.markdown("---")
st.markdown("## 📊 Mood History (Last Sessions)")

mood_counts = get_mock_mood_history()
if mood_counts:
    fig, ax = plt.subplots()
    moods = list(mood_counts.keys())
    counts = list(mood_counts.values())
    ax.bar(moods, counts, color='skyblue')
    ax.set_title("Mood Frequency Chart")
    st.pyplot(fig)
else:
    st.info("No mood data available yet.")
