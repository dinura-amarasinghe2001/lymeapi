import os
import zipfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf

app = FastAPI(title="Lyme Disease Detection API")

# --- Google Drive model zip ---
MODEL_FOLDER = "model"
MODEL_ZIP = "model.zip"
FILE_ID = "1-1g5H7G_lj3riz4WrcEHP_JM28WsAKO4"

# Download & extract if not exists
if not os.path.exists(MODEL_FOLDER):
    import gdown
    print("Downloading models from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_ZIP, quiet=False)

    print("Extracting models...")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")  # extract to current dir
    print("Models ready!")

# --- Confirm TFLite files exist ---
MODEL_PATH = os.path.join(MODEL_FOLDER, "kfold_synthetic_adaptiveenhancement")
for i in range(1, 6):
    path = os.path.join(MODEL_PATH, f"fold_{i}", "model.tflite")
    if not os.path.exists(path):
        raise FileNotFoundError(f"TFLite model not found: {path}")
print("All TFLite files found!")

# --- Load TFLite interpreters ---
interpreters = []
for i in range(1, 6):
    path = os.path.join(MODEL_PATH, f"fold_{i}", "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    interpreters.append(interpreter)
print("TFLite interpreters ready!")

# --- Class names ---
class_names = ['em', 'pityriasis', 'ringworm']
IMG_SIZE = (224, 224)

# --- Adaptive enhancement function ---
def adaptive_enhance(img, min_contrast=0.3, max_clip=4.0):
    img = img.astype(np.float32) / 255.0
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    contrast = np.std(L) / 255.0
    clip = max(1.0, max_clip * (1 - contrast))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    L_final = cv2.addWeighted(L, 0.5, L_clahe, 0.5, 0)
    lab_final = cv2.merge((L_final, A, B))
    enhanced = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)
    gamma = 1.4 * (1 - contrast)
    enhanced = np.power(enhanced.astype(np.float32) / 255.0, 1 / gamma)
    return np.clip(enhanced, 0, 1)

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_enh = adaptive_enhance(img)
    img_input = np.expand_dims(img_enh, axis=0).astype(np.float32)

    # Ensemble prediction using TFLite interpreters
    preds = []
    for interpreter in interpreters:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
        preds.append(pred)
    avg_preds = np.mean(preds, axis=0) * 100

    result = {cls: round(float(prob), 2) for cls, prob in zip(class_names, avg_preds)}
    return JSONResponse(content=result)
