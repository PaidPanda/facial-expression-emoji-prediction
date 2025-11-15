from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('my_emotion_recognizer_best_69.4%.keras')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=[0,-1])
        pred = model.predict(face, verbose=0)[0]
        label = EMOTIONS[np.argmax(pred)]
        conf = pred.max()
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    cv2.imshow('Facial Emotion Recognition - Press Q to quit', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()