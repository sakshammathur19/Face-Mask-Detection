import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("😷 Face Mask Detection")

model = load_model("mask_detector.keras", compile=False)
labels = ["Mask", "No Mask"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict(face):
    face = cv2.resize(face, (100, 100))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face, verbose=0)[0]
    idx = np.argmax(pred)

    return labels[idx], float(pred[idx])

mode = st.radio("Choose Mode", ["Upload", "Live Camera (Frame)"])

# UPLOAD
if mode == "Upload":
    file = st.file_uploader("Upload Image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.2, 5)

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]

            label, conf = predict(face)

            color = (0,255,0) if label=="Mask" else (0,0,255)

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, f"{label} {conf*100:.1f}%",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# LIVE (SAFE STREAMLIT VERSION)
if mode == "Live Camera (Frame)":
    cam = st.camera_input("Take Photo")

    if cam:
        img = cv2.imdecode(np.frombuffer(cam.read(), np.uint8), 1)

        faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.2, 5)

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]

            label, conf = predict(face)

            color = (0,255,0) if label=="Mask" else (0,0,255)

            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img, f"{label} {conf*100:.1f}%",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))