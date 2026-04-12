import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


model = load_model("mask_detector.keras")


labels = ['Mask', 'No Mask']  

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
     
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        try:
        
            face = cv2.resize(face, (100, 100))  
        except:
            continue

  
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        
        pred = model.predict(face, verbose=0)[0]
        confidence = np.max(pred)

        if confidence > 0.6:
            label = labels[np.argmax(pred)]
        else:
            label = "Uncertain"

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        
        cv2.putText(image, f"{label} ({confidence*100:.1f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    return image



st.set_page_config(page_title="Face Mask Detection", layout="centered")
st.title("Face Mask Detection")
st.write("Detect whether a person is wearing a mask using your webcam or uploaded image.")

option = st.radio("Choose input source:", ('Upload Image', 'Use Webcam'))

if option == 'Upload Image':
    uploaded = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'])
    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        result = detect_mask(image_bgr)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Image", use_column_width=True)

elif option == 'Use Webcam':
    

    run = st.button("Start Webcam Detection")
    if run:
        cap = cv2.VideoCapture(0)
        st.info("Press 'Q' in the webcam window to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break
            output = detect_mask(frame)
            cv2.imshow("Webcam - Mask Detection", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
