import os
from collections import Counter
import shutil
from sklearn.model_selection import train_test_split


def prepare_dataset(
        dataset_path:str = '../DataFiles',
        emotions=None,
        train_dir = "../DataSets/train",
        val_dir = "../DataSets/val",
        test_dir = "../DataSets/test"):
    if emotions is None:
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Load file paths and labels
    image_paths = []
    labels = []
    for inx, emotion in enumerate(emotions):
        emotion_path = os.path.join(dataset_path, emotion)
        for img_name in os.listdir(emotion_path):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                image_paths.append(os.path.join(emotion_path, img_name))
                labels.append(inx)
    print(f'Total images found: {len(image_paths)}')
    print("Class distribution:")
    print(dict(Counter([emotions[label] for label in labels])))

    # Split dataset into training and testing sets

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
    def create_split_folders(paths, labels, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        for emotion in emotions:
            os.makedirs(f"{folder_name}/{emotion}", exist_ok=True)

        for path, label in zip(paths, labels):
            emotion = emotions[label]
            dest = f"{folder_name}/{emotion}/{os.path.basename(path)}"
            shutil.copy(path, dest)

    print("Creating folders...")
    create_split_folders(X_train, y_train, train_dir)
    create_split_folders(X_val, y_val, val_dir)
    create_split_folders(X_test, y_test, test_dir)
    print("Folders created successfully.")
