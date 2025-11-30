# import base64, io
# import numpy as np
# from PIL import Image

# def decode_and_preprocess(img_base64):
#     img_bytes = base64.b64decode(img_base64)
#     img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
#     img = img.resize((48,48))
#     x = np.array(img).astype("float32") / 255.0
#     x = np.expand_dims(x, axis=[0,-1])
#     return x

import base64, io
import numpy as np
import cv2
from PIL import Image
import os

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def decode_and_preprocess(img_base64):
    # Decode base64 → PIL image
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Convert to OpenCV format
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # If face found → crop detected face
        (x, y, w, h) = faces[0]
        face_region = gray[y:y+h, x:x+w]
    else:
        # No face → center fallback crop
        h, w = gray.shape
        size = min(h, w)
        startx = w//2 - size//4
        starty = h//2 - size//4
        face_region = gray[starty:starty + size//2, startx:startx + size//2]

    # Resize to model size
    face_region = cv2.resize(face_region, (48, 48))

    # Normalize + reshape like training data
    x = face_region.astype("float32") / 255.0
    x = np.expand_dims(x, axis=[0, -1])  # shape = (1,48,48,1)

    return x
