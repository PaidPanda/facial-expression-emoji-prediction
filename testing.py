from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --------------------------
# LOAD BEST SAVED MODEL
# --------------------------
print("Loading best model...")
model = load_model("Models/training_efficientnet_b0.keras")

# --------------------------
# LOAD TEST DATASET
# --------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'DataSets/test/',
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --------------------------
# EVALUATE
# --------------------------
print("Evaluating...")
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"\nTEST ACCURACY = {acc*100:.2f}%")
print(f"TEST LOSS     = {loss:.4f}")