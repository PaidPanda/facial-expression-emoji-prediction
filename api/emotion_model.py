import numpy as np
from tensorflow.keras.models import load_model
from config import EMOTIONS, MODEL_PATH

class EmotionModel:
    def __init__(self):
        self.model = load_model(MODEL_PATH)

    def predict(self, preprocessed_image):
        pred = self.model.predict(preprocessed_image, verbose=0)[0]
        label = EMOTIONS[int(np.argmax(pred))]
        confidence = float(pred.max() * 100)
        return label, confidence