import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the model
model = load_model('Models/training_mobilenet_v2.keras')

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face region (still grayscale)
        face = gray[y:y + h, x:x + w]

        # Resize to model's expected input size: 96x96
        face = cv2.resize(face, (224, 224))

        # Normalize to [0, 1]
        face = face.astype('float32') / 255.0

        # Convert grayscale to RGB by duplicating the channel (since model expects 3 channels)
        face = np.stack((face,) * 3, axis=-1)  # Shape: (96, 96, 3)

        # Add batch dimension: (1, 96, 96, 3)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        pred = model.predict(face, verbose=0)[0]
        label = EMOTIONS[np.argmax(pred)]
        conf = pred.max()

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Facial Emotion Recognition - Press Q to quit', frame)

    # Quit on 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()