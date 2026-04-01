import cv2
import os
import streamlit as st
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time

# Charger le modèle
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def process_frame(frame, color, min_neighbors, scale_factor, save_faces):
    global count   # ou utilise st.session_state.face_count comme avant
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 480))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        if save_faces:
            face_img = img[y:y+h, x:x+w]
            os.makedirs("faces", exist_ok=True)
            count = st.session_state.get("face_count", 0)
            cv2.imwrite(f"faces/face_{count}.jpg", face_img)
            st.session_state.face_count = count + 1

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def app():
    st.title("📸 Détection de visages (Streamlit + OpenCV)")

    color_hex = st.color_picker("Couleur du rectangle", "#00FF00")
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))

    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.05)
    save_faces = st.checkbox("Sauvegarder les visages détectés", value=False)

    if "face_count" not in st.session_state:
        st.session_state.face_count = 0
    if "webrtc_key" not in st.session_state:
        st.session_state.webrtc_key = f"face-det-{int(time.time())}"

    # === NOUVELLE CONFIGURATION RTC (plus robuste) ===
    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                # TURN Twilio (gratuit pour usage modéré - crée un compte Twilio si besoin)
                # {"urls": "turn:global.turn.twilio.com:3478?transport=udp",
                #  "username": "ton_username_twilio",
                #  "credential": "ton_credential_twilio"},
            ],
            "iceTransportPolicy": "all",   # ou "relay" si tu utilises seulement TURN
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Démarrer la détection", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏹️ Arrêter la détection", use_container_width=True):
            st.session_state.running = False
            st.rerun()

    if st.session_state.get("running", False):
        webrtc_ctx = webrtc_streamer(
            key=st.session_state.webrtc_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=lambda frame: process_frame(
                frame, color, min_neighbors, scale_factor, save_faces
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.success("✅ Détection en cours - Webcam connectée")

    # Affichage des images sauvegardées (comme avant)
    st.subheader("📂 Images sauvegardées")
    if os.path.exists("faces") and os.listdir("faces"):
        cols = st.columns(4)
        for i, file in enumerate(sorted(os.listdir("faces"))):
            try:
                img = Image.open(os.path.join("faces", file))
                with cols[i % 4]:
                    st.image(img, caption=file, use_container_width=True)
            except:
                pass
    else:
        st.info("Aucune image sauvegardée pour le moment.")

    if st.button("🗑️ Effacer toutes les images"):
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
