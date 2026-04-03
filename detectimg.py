import cv2
import streamlit as st
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

# Charger le modèle
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 🎯 Classe WebRTC (OBLIGATOIRE)
class FaceDetector(VideoTransformerBase):
    def __init__(self):
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=st.session_state.scale_factor,
            minNeighbors=st.session_state.min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                img,
                (x, y),
                (x+w, y+h),
                st.session_state.color,
                2
            )

            if st.session_state.save_faces:
                if not os.path.exists("faces"):
                    os.makedirs("faces")

                face_img = img[y:y+h, x:x+w]
                cv2.imwrite(f"faces/face_{self.count}.jpg", face_img)
                self.count += 1

        return img


def app():
    st.title("📸 Détection de visages (Viola-Jones)")

    # 📘 Instructions
    st.markdown("""
    ### 📌 Instructions :
    1. Cliquez sur **Start** pour activer la webcam  
    2. Ajustez les paramètres (minNeighbors, scaleFactor)  
    3. Choisissez une couleur pour les rectangles  
    4. Activez la sauvegarde si nécessaire  
    """)

    # 🎨 Couleur
    color_hex = st.color_picker("Couleur du rectangle", "#00FF00")
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))

    # 🎚 Paramètres
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    save_faces = st.checkbox("💾 Sauvegarder les visages")

    # 🔁 Stocker dans session_state
    st.session_state.color = color
    st.session_state.min_neighbors = min_neighbors
    st.session_state.scale_factor = scale_factor
    st.session_state.save_faces = save_faces

    # 🎥 Webcam
    webrtc_streamer(
        key="face-detection",
        video_transformer_factory=FaceDetector
    )

    # 📂 Affichage images
    if os.path.exists("faces"):
        st.subheader("📂 Images sauvegardées")

        for file in os.listdir("faces"):
            img = Image.open(os.path.join("faces", file))
            st.image(img, caption=file)


if __name__ == "__main__":
    app()
