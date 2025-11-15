from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('my_emotion_recognizer_best_69.4%.keras')
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image_path):
    img = cv2.imread(image_path, 0)  # grayscale
    img = cv2.resize(img, (48,48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0,-1])
    pred = model.predict(img, verbose=0)[0]
    label = EMOTIONS[np.argmax(pred)]
    confidence = pred.max()
    return label, confidence

label, conf = predict_emotion('test_image.jpg')
print(f"Emotion: {label.upper()} ({conf:.2%})")