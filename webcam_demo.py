from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import deque

model = load_model("my_emotion_recognizer_best_69.4%.keras")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------------------------------------
# LOAD PNG EMOJIS (ONLY FOR EMOTIONS YOU HAVE FILES FOR)
# -----------------------------------------------------------

def load_emoji_set(prefix):
    """Loads the 4-level emoji set for one emotion."""
    return [
        cv2.imread(f"emojis/{prefix}_1.png", cv2.IMREAD_UNCHANGED),
        cv2.imread(f"emojis/{prefix}_2.png", cv2.IMREAD_UNCHANGED),
        cv2.imread(f"emojis/{prefix}_3.png", cv2.IMREAD_UNCHANGED),
        cv2.imread(f"emojis/{prefix}_4.png", cv2.IMREAD_UNCHANGED),
    ]

emoji_sets = {
    "happy": load_emoji_set("happy"),
    "angry": load_emoji_set("angry"),
    "disgust": load_emoji_set("disgust"),
    "fear": load_emoji_set("fear"),        
    "surprise": load_emoji_set("surprise"),
    "neutral": load_emoji_set("neutral"),
    "sad": load_emoji_set("sad")
}

# -----------------------------------------------------------
# PNG OVERLAY FUNCTION (works with alpha channel)
# -----------------------------------------------------------

def overlay_emoji(frame, emoji, x, y, w):
    emoji = cv2.resize(emoji, (w, w))
    eh, ew = emoji.shape[:2]

    y1 = max(0, y - eh)
    y2 = y1 + eh
    x1 = x
    x2 = x + ew

    if y2 > frame.shape[0]: y2 = frame.shape[0]
    if x2 > frame.shape[1]: x2 = frame.shape[1]

    emoji = emoji[0:y2-y1, 0:x2-x1]
    b,g,r,a = cv2.split(emoji)
    alpha = a / 255.0

    roi = frame[y1:y2, x1:x2]

    for c in range(3):
        roi[:, :, c] = (alpha * emoji[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)

    frame[y1:y2, x1:x2] = roi
    return frame

# -----------------------------------------------------------
# SMOOTH PREDICTIONS (avoid flicker)
# -----------------------------------------------------------
history = deque(maxlen=8)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=[0,-1])

        pred = model.predict(face, verbose=0)[0]
        label = EMOTIONS[np.argmax(pred)]
        confidence = pred.max() * 100

        # smoothing
        history.append(label)
        stable_label = max(set(history), key=history.count)

        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # ---------------------------------------------------
        # DISPLAY EMOJI IF PNG AVAILABLE
        # ---------------------------------------------------
        if stable_label in emoji_sets:
            if confidence <= 25: idx = 0
            elif confidence <= 50: idx = 1
            elif confidence <= 75: idx = 2
            else: idx = 3

            emoji = emoji_sets[stable_label][idx]
            frame = overlay_emoji(frame, emoji, x, y, w)

        else:
            # fallback text if no PNG for that emotion yet
            cv2.putText(frame, f"{stable_label} {confidence:.1f}%",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

    cv2.imshow("Emoji Emotion Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
