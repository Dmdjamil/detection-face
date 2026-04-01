import cv2
import streamlit as st
import os
from PIL import Image

# Charger le modèle Haarcascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(color, min_neighbors, scale_factor, save_faces):
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])  # zone d'affichage
    count = 0

    run = st.session_state.get("run", True)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur webcam")
            break

        # 🔥 Accélération (resize)
        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            if save_faces:
                face_img = frame[y:y+h, x:x+w]
                if not os.path.exists("faces"):
                    os.makedirs("faces")
                cv2.imwrite(f"faces/face_{count}.jpg", face_img)
                count += 1

        # ⚠️ Convertir BGR → RGB pour Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_window.image(frame)

    cap.release()
    
#Afficher les images capturer
if os.path.exists("faces"):
    st.subheader("📂 Images sauvegardées")

    for file in os.listdir("faces"):
        img = Image.open(os.path.join("faces", file))
        st.image(img, caption=file)

def app():
    st.title("📸 Détection de visages (Streamlit + OpenCV)")

    # 🎨 Couleur
    color_hex = st.color_picker("Couleur du rectangle", "#00FF00")
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))

    # 🎚 Paramètres
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # 💾 Sauvegarde
    save_faces = st.checkbox("Sauvegarder les visages")

    # ▶ Boutons
    if "run" not in st.session_state:
        st.session_state.run = False

    if st.button("Démarrer"):
        st.session_state.run = True
        detect_faces(color, min_neighbors, scale_factor, save_faces)

    if st.button("Arrêter"):
        st.session_state.run = False


if __name__ == "__main__":
    app()
