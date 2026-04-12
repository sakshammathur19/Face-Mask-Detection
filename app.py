import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------
# LOAD MODEL
# -----------------------
model = load_model("mask_detector.keras", compile=False)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("😷 Face Mask Detection (Upload + Live)")

mode = st.sidebar.radio("Select Mode", ["📤 Upload Image", "🎥 Live Camera"])

# -----------------------
# PREDICT FUNCTION
# -----------------------
def predict(face):

    face = cv2.resize(face, (100, 100))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face, verbose=0)[0]

    if len(pred) == 1:
        return ("No Mask", pred[0]) if pred[0] > 0.5 else ("Mask", 1 - pred[0])

    return ("Mask", pred[0]) if pred[0] > pred[1] else ("No Mask", pred[1])


# -----------------------
# UPLOAD MODE
# -----------------------
if mode == "📤 Upload Image":

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file is not None:

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            label, conf = predict(face)

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            cv2.putText(img,
                        f"{label} {conf*100:.1f}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# -----------------------
# LIVE CAMERA MODE
# -----------------------
elif mode == "🎥 Live Camera":

    img = st.camera_input("Take a picture")

    if img is not None:

        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            label, conf = predict(face)

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(frame,
                        f"{label} {conf*100:.1f}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))