import os
import cv2
import numpy as np
import pandas as pd
import gdown
import zipfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

app = FastAPI(title="Lyme Disease Detection API")



MODEL_FOLDER = "model"
MODEL_ZIP = "model.zip"


FILE_ID = "1ROOVIWiVis3bHK9C16JHvcAAu2z6aKMT"

if not os.path.exists(MODEL_FOLDER):
    print("Downloading models from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_ZIP, quiet=False)

    print("Extracting models...")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall()
    print("Models ready")



MODEL_PATH = os.path.join(MODEL_FOLDER, "kfold_synthetic_adaptiveenhancement")
MODEL_PATHS = [
    os.path.join(MODEL_PATH, f"fold_{i}", "model.h5") for i in range(1, 6)
]

print("Loading models...")
models = [load_model(path) for path in MODEL_PATHS]
print("Models loaded successfully!")

# Class names
class_names = ['em', 'pityriasis', 'ringworm']

# Image size for model
IMG_SIZE = (224, 224)


# Adaptive enhancement function


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


# Prediction endpoint


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # Apply enhancement
    img_enh = adaptive_enhance(img)
    img_input = np.expand_dims(img_enh, axis=0)

    # Ensemble prediction (average over all models)
    preds = np.array([model.predict(img_input)[0] for model in models])
    avg_preds = np.mean(preds, axis=0)
    avg_preds_percent = avg_preds * 100

    # Format results
    result = {cls: round(float(prob), 2) for cls, prob in zip(class_names, avg_preds_percent)}

    return JSONResponse(content=result)

