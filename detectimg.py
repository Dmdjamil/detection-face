import cv2
import os
import streamlit as st
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import time

# Charger le modèle une seule fois
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class FaceDetector(VideoTransformerBase):
    def __init__(self, color, min_neighbors, scale_factor, save_faces):
        self.color = color
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        self.save_faces = save_faces

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))   # Accélération

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 2)

            if self.save_faces:
                face_img = img[y:y+h, x:x+w]
                os.makedirs("faces", exist_ok=True)
                count = st.session_state.get("face_count", 0)
                cv2.imwrite(f"faces/face_{count:04d}.jpg", face_img)
                st.session_state.face_count = count + 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def app():
    st.title("📸 Détection de visages en temps réel")

    # Paramètres utilisateur
    color_hex = st.color_picker("Couleur du rectangle", "#00FF00")
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))   # BGR

    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.05)
    save_faces = st.checkbox("Sauvegarder les visages détectés", False)

    if "face_count" not in st.session_state:
        st.session_state.face_count = 0

    # Clé unique pour éviter les conflits
    if "webrtc_key" not in st.session_state:
        st.session_state.webrtc_key = f"face-detection-{int(time.time())}"

    # Configuration RTC (plusieurs STUN pour meilleure connexion)
    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        }
    )

    # Boutons de contrôle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Démarrer la webcam", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏹️ Arrêter la webcam", use_container_width=True):
            st.session_state.running = False
            st.rerun()

    # Lancement du streamer uniquement si demandé
    if st.session_state.get("running", False):
        ctx = webrtc_streamer(
            key=st.session_state.webrtc_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_transformer_factory=lambda: FaceDetector(color, min_neighbors, scale_factor, save_faces),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.state.playing:
            st.success("✅ Webcam connectée - Détection active")

    # Affichage des visages sauvegardés
    st.subheader("📂 Visages sauvegardés")
    if os.path.exists("faces") and os.listdir("faces"):
        cols = st.columns(4)
        for i, filename in enumerate(sorted(os.listdir("faces"))):
            try:
                with cols[i % 4]:
                    st.image(os.path.join("faces", filename), caption=filename, use_container_width=True)
            except:
                pass
    else:
        st.info("Aucune image sauvegardée pour l'instant.")

    # Bouton nettoyage
    if st.button("🗑️ Supprimer toutes les images"):
        if os.path.exists("faces"):
            for f in os.listdir("faces"):
                os.remove(os.path.join("faces", f))
        st.session_state.face_count = 0
        st.success("Images supprimées")
        st.rerun()


if __name__ == "__main__":
    if "running" not in st.session_state:
        st.session_state.running = False
    app()
