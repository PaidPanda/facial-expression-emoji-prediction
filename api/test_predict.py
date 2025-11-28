import base64
import os
import requests
from dotenv import load_dotenv
from config import TEST_IMAGE

load_dotenv()
url = os.getenv("ML_ENDPOINT")

with open(TEST_IMAGE, "rb") as f:
    img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

payload = {"image": img_base64}
response = requests.post(url, json=payload)
print(response.json())
