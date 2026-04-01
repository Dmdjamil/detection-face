import cv2
import os
import streamlit as st
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Charger le modèle Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Variable globale pour le compteur (sauvegarde des visages)
count = 0

def video_frame_callback(frame, color, min_neighbors, scale_factor, save_faces):
    """Fonction appelée sur chaque frame de la webcam"""
    global count
    
    img = frame.to_ndarray(format="bgr24")   # Format OpenCV

    # Accélération : resize
    img = cv2.resize(img, (640, 480))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    for (x, y, w, h) in faces:
        # Dessiner le rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Sauvegarder le visage si l'option est activée
        if save_faces:
            face_img = img[y:y + h, x:x + w]
            if not os.path.exists("faces"):
                os.makedirs("faces")
            cv2.imwrite(f"faces/face_{count}.jpg", face_img)
            count += 1

    # Retourner la frame traitée
    return av.VideoFrame.from_ndarray(img, format="bgr24")


def app():
    st.title("📸 Détection de visages (Streamlit + OpenCV)")

    # === Paramètres ===
    color_hex = st.color_picker("Couleur du rectangle", "#00FF00")
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))   # BGR

    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.05)

    save_faces = st.checkbox("Sauvegarder les visages détectés", value=False)

    # Configuration WebRTC (obligatoire pour Streamlit Cloud)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # === Bouton Démarrer ===
    if st.button("▶️ Démarrer la détection"):
        st.session_state.running = True

    if st.button("⏹️ Arrêter la détection"):
        st.session_state.running = False

    # === Lancement du flux vidéo ===
    if st.session_state.get("running", False):
        webrtc_ctx = webrtc_streamer(
            key="face-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=lambda frame: video_frame_callback(
                frame, color, min_neighbors, scale_factor, save_faces
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.success("✅ Détection en cours...")

    # === Affichage des visages sauvegardés ===
    st.subheader("📂 Images sauvegardées")
    if os.path.exists("faces") and len(os.listdir("faces")) > 0:
        cols = st.columns(3)  # Affichage en grille
        for i, file in enumerate(sorted(os.listdir("faces"))):
            try:
                img = Image.open(os.path.join("faces", file))
                with cols[i % 3]:
                    st.image(img, caption=file, use_container_width=True)
            except Exception:
                st.warning(f"Impossible d'afficher {file}")
    else:
        st.info("Aucune image sauvegardée pour le moment.")

if __name__ == "__main__":
    # Initialisation de la session
    if "running" not in st.session_state:
        st.session_state.running = False
    
    app()
