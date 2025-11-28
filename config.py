import os

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "Models/my_emotion_recognizer_best_69.4%.keras"
HISTORY_SIZE = 8
IMAGE_SIZE = (48,48)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(PROJECT_ROOT, 'DataFiles', 'angry', 'angry_00000.jpg')
