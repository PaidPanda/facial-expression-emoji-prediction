# Facial Emotion Recognition

## Overview

This project uses a Convolutional Neural Network (CNN) to recognize human emotions from facial images or live webcam feeds.
It supports:

* Training and evaluating multiple CNN models
* Real-time emotion detection via webcam
* Deployment on NVIDIA Jetson devices using TensorRT (`.engine` model)

---

## Features

* Train models on your own dataset
* Predict emotions from images or webcam feed
* Compare multiple models for performance
* Save trained models in `.h5` and `.keras` formats
* Visualize results and confusion matrices

---

## Requirements

* Python 3.10+
* Webcam (for live demo)
* Dependencies listed in `requirements.txt`

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure the dataset folder `DataFiles` is in the project root:

```
DataFiles/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

### 3. Train Models

Run `train.bat` (double-click) and choose one of the options:

1. Run 4 comparison models
2. Run the best CNN model
3. Run webcam demo (ensure 'emojis' folder is in project root)
4. Exit

### 4. Run Webcam Demo

Ensure `my_emotion_recognizer_best_69.4%.keras` is in the root folder and run:

```bash
python webcam_demo.py
```

Press `Q` to quit.

### 5. Run Application on Jetson

Ensure `emotionnet.engine` is in the project folder and run:

```bash
python jetson_demo.py
```

### 6. Compare Models (Train/Test)

* Navigate to the `training_models` directory
* Run:

```bash
python main.py
```

* Follow prompts:

  1. Input `1` for data injection and dataset preparation
  2. Wait for dataset to download
  3. Select model of choice

* View output in `Models/` and `Graphs/` folders

### 7. Compute Final Model

* Ensure `main.py` option `1` has already run for dataset preparation
* Run:

```bash
python cNN.py
```

* This will generate `my_emotion_recognizer.h5` in the root folder
* Convert `.h5` to `.keras` format:

```bash
python convert_h5_to_keras.py
```

* Run the webcam demo again if desired:

```bash
python webcam_demo.py
```

---

## Directory Structure

```
project_root/
│
├── DataFiles/                 # Dataset folders
├── cNN.py                     # CNN training script
├── convert_h5_to_keras.py
├── webcam_demo.py
├── inference.py
├── train.bat                  # Shortcut for training
├── requirements.txt
└── README.md
```

---

## Notes

* For Jetson deployment, use `emotionnet.engine` with `jetson_demo.py`
* Model files should remain in the project root for scripts to access them
* Training may take several minutes depending on hardware
* See `instructions.txt` for detailed step-by-step setup
