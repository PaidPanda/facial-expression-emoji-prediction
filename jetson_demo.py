import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import time

# ----------------------------------------------------
# TENSORRT ENGINE LOADING
# ----------------------------------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

print("[INFO] Loading TensorRT engine...")
engine = load_engine("emotionnet.engine")
context = engine.create_execution_context()
print("[INFO] Engine loaded!")

# ----------------------------------------------------
# BUFFER ALLOCATION (with PyCUDA)
# ----------------------------------------------------
def allocate_buffers(engine):
    bindings = []
    host_inputs = []
    host_outputs = []
    device_inputs = []
    device_outputs = []

    for binding in engine:
        idx = engine.get_binding_index(binding)

        dtype = trt.nptype(engine.get_binding_dtype(idx))
        shape = engine.get_binding_shape(idx)
        size = trt.volume(shape)

        # Host buffer
        host_mem = np.zeros(size, dtype=dtype)

        # Device buffer
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            device_inputs.append(device_mem)
        else:
            host_outputs.append(host_mem)
            device_outputs.append(device_mem)

    return host_inputs, host_outputs, device_inputs, device_outputs, bindings

host_inputs, host_outputs, device_inputs, device_outputs, bindings = allocate_buffers(engine)

input_shape = engine.get_binding_shape(0)  # (1,1,48,48)
_, C, H, W = input_shape

# ----------------------------------------------------
# EMOTION LABELS
# ----------------------------------------------------
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ----------------------------------------------------
# FACE DETECTOR
# ----------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

# ----------------------------------------------------
# INFERENCE FUNCTION
# ----------------------------------------------------
def predict_emotion(face_img):

    # Preprocessing
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    resized = resized.astype(np.float32) / 255.0

    # Convert to NCHW
    chw = resized.reshape(1, 1, 48, 48)

    # Copy to host input buffer
    np.copyto(host_inputs[0], chw.ravel())

    # Host → Device
    cuda.memcpy_htod(device_inputs[0], host_inputs[0])

    # Run inference
    context.execute_v2(bindings)

    # Device → Host
    cuda.memcpy_dtoh(host_outputs[0], device_outputs[0])

    # Softmax
    out = host_outputs[0]
    probs = np.exp(out - np.max(out))
    probs = probs / np.sum(probs)

    label_id = int(np.argmax(probs))
    return EMOTIONS[label_id], float(probs[label_id])

# ----------------------------------------------------
# WEBCAM LOOP
# ----------------------------------------------------
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        label, conf = predict_emotion(face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2,
        )

    cv2.imshow("TensorRT Emotion Recognition (PyCUDA)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()