import os
import numpy as np
from collections import Counter

DATASET_PATH = 'DataFiles'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

image_paths = []
labels = []

for inx, emotion in enumerate(EMOTIONS):
    emotion_path = os.path.join(DATASET_PATH, emotion)
    for img_name in os.listdir(emotion_path):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            image_paths.append(os.path.join(emotion_path, img_name))
            labels.append(inx)

print(f'Total images found: {len(image_paths)}')
print("Class distribution:")
print(dict(Counter([EMOTIONS[label] for label in labels])))

# Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# split into train (80%) and temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# split temp into val (10%) and test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Create train, val, test folders
import shutil

def create_split_folders(paths, labels, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for emotion in EMOTIONS:
        os.makedirs(f"{folder_name}/{emotion}", exist_ok=True)

    for path, label in zip(paths, labels):
        emotion = EMOTIONS[label]
        dest = f"{folder_name}/{emotion}/{os.path.basename(path)}"
        shutil.copy(path, dest)

print("Creating folders...")
create_split_folders(X_train, y_train, 'train')
create_split_folders(X_val, y_val, 'val')
create_split_folders(X_test, y_test, 'test')
print("Folders created successfully.")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

BATCH_SIZE = 64
IMG_SIZE = (48, 48)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'train/',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    'val/',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    'test/',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("Class weights applied:")
for i, emo in enumerate(EMOTIONS):
    print(f"  {emo:8} → {class_weight_dict[i]:.2f}x")

# Train a CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1), padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
], name="EmotionNet")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5),
    ModelCheckpoint('best_emotion_model.h5', save_best_only=True)
]

print("Training started...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# --------------------------------------------------------------
# STEP 5: FINAL TEST + CONFUSION MATRIX
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")

# Predictions
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_gen.classes

# Report
print("\n" + "="*50)
print(classification_report(y_true, y_pred, target_names=EMOTIONS))
print("="*50)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Save final model
model.save('my_emotion_recognizer.h5')
print("Model saved as 'my_emotion_recognizer.h5'")