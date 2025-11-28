import base64, io
import numpy as np
from PIL import Image

def decode_and_preprocess(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
    img = img.resize((48,48))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=[0,-1])
    return x
