import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


model = load_model("mask_detector.keras")
labels = ['Mask', 'No Mask']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


prev_label = ""
label_count = 0
threshold_frames = 5  
display_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    small_frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = small_frame[y:y+h, x:x+w]

        try:
            face_input = cv2.resize(face, (100, 100))
            face_input = face_input.astype("float") / 255.0
            face_input = img_to_array(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            
            prediction = model.predict(face_input, verbose=0)[0]
            label = labels[np.argmax(prediction)]
            confidence = float(np.max(prediction))

           
            if label == prev_label:
                label_count += 1
            else:
                label_count = 1
                prev_label = label

            if label_count >= threshold_frames:
                display_label = label

            
            color = (0, 255, 0) if display_label == "Mask" else (0, 0, 255)
            cv2.putText(small_frame, f"{display_label} ({confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(small_frame, (x, y), (x + w, y + h), color, 2)

        except Exception as e:
            print(f"Error processing face: {e}")

    
    cv2.imshow("Face Mask Detection", small_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
