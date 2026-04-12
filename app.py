import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("mask_detector.keras")
labels = ['Mask', 'No Mask']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_mask(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (100, 100))
        except:
            continue

        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0]
        label = labels[np.argmax(pred)]

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

interface = gr.Interface(
    fn=detect_mask,

    inputs=gr.Image(source="webcam", type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Face Mask Detection",
    description="Upload an image to detect mask"
)

interface.launch()