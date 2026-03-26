import json
import numpy as np
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================
# CONFIG
# ==============================
IMG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.50
MAX_IMAGE_SIZE_MB = 5

MODEL_PATH   = "models/modelo_ml_savedmodel"
CLASSES_PATH = "models/classes.json"


# ==============================
# CARREGAMENTO DO MODELO
# ==============================
print("Carregando modelo...")

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Detecta automaticamente os nomes corretos de entrada e saída
input_key  = list(infer.structured_input_signature[1].keys())[0]
output_key = list(infer.structured_outputs.keys())[0]

print(f"Chave de entrada:  '{input_key}'")
print(f"Chave de saída:    '{output_key}'")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

print("Modelo carregado com sucesso!")
print("Classes:", classes)


# ==============================
# APP
# ==============================
app = FastAPI(
    title="API - Detecção de Doenças na Soja",
    description="Classifica doenças em plantações de soja usando EfficientNetB3",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Imagem inválida ou corrompida")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return tf.constant(img, dtype=tf.float32)


def confidence_level(conf: float) -> str:
    if conf > 0.80:
        return "Alta"
    if conf > 0.60:
        return "Média"
    return "Baixa"


# ==============================
# ENDPOINTS
# ==============================
@app.get("/")
def root():
    return {
        "status": "online",
        "modelo": "EfficientNetB3",
        "classes": classes,
        "uso": "POST /predict com uma imagem"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Imagem muito grande. Máximo: {MAX_IMAGE_SIZE_MB}MB"
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Arquivo inválido. Envie uma imagem (jpg, png, etc.)"
        )

    try:
        img_tensor = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Inferência passando o tensor com o nome correto da chave de entrada
    result      = infer(**{input_key: img_tensor})
    predictions = result[output_key].numpy()[0]

    index      = int(np.argmax(predictions))
    confidence = float(predictions[index])
    disease    = classes[index]

    probabilities = {
        classes[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(predictions))
    }
    probabilities = dict(
        sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    )

    top3 = list(probabilities.items())[:3]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "resultado":       "Inconclusivo",
            "confianca":       round(confidence * 100, 2),
            "nivel_confianca": confidence_level(confidence),
            "top3":            top3,
            "probabilidades":  probabilities
        }

    return {
        "resultado":       disease,
        "confianca":       round(confidence * 100, 2),
        "nivel_confianca": confidence_level(confidence),
        "top3":            top3,
        "probabilidades":  probabilities
    }