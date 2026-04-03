import cv2
import streamlit as st
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Charger Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Classe de traitement vidéo
class FaceDetector(VideoTransformerBase):
    def __init__(self):
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        return img


def app():
    st.title("📸 Détection de visages")

    webrtc_streamer(
        key="face-detection",
        video_transformer_factory=FaceDetector
    )

    # Afficher images sauvegardées (optionnel)
    if os.path.exists("faces"):
        st.subheader("📂 Images sauvegardées")
        for file in os.listdir("faces"):
            img = Image.open(os.path.join("faces", file))
            st.image(img, caption=file)


if __name__ == "__main__":
    app()
