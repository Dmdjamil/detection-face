import cv2
import streamlit as st
import os
import numpy as np
from PIL import Image

# Charger modèle
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def app():
    st.title("📸 Détection de visages")

    # 📸 Capture image
    img_file = st.camera_input("Prendre une photo")

    # 🎨 Couleur
    color_hex = st.color_picker("Couleur du rectangle", "#00FF00")
    color = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))

    # 🎚 Paramètres
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # 💾 Sauvegarde
    save_faces = st.checkbox("Sauvegarder les visages")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            if save_faces:
                face_img = frame[y:y+h, x:x+w]
                if not os.path.exists("faces"):
                    os.makedirs("faces")
                cv2.imwrite(f"faces/face_{i}.jpg", face_img)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame)

    # 📂 Afficher images
    if os.path.exists("faces"):
        st.subheader("📂 Images sauvegardées")
        for file in os.listdir("faces"):
            img = Image.open(os.path.join("faces", file))
            st.image(img, caption=file)


if __name__ == "__main__":
    app()
