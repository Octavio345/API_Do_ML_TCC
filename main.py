import io
import json
import numpy as np
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================
# CONFIG
# ==============================
IMG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.18
MIN_VEGETATION_RATIO = 0.04
MIN_SHARPNESS = 18.0
MAX_IMAGE_SIZE_MB = 5

MODEL_PATH   = "models/modelo_ml_savedmodel"
CLASSES_PATH = "models/classes.json"

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
def read_image_rgb(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return np.asarray(image)
    except (UnidentifiedImageError, OSError, ValueError):
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagem invalida ou corrompida")

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyze_quality_and_domain(rgb_image: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    vegetation_mask = (
        (hue >= 18)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 35)
    )

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    return {
        "vegetation_ratio": round(float(np.mean(vegetation_mask)), 4),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness": round(float(np.mean(gray)), 2),
    }


def is_out_of_domain(quality: dict) -> tuple[bool, str | None]:
    if quality["vegetation_ratio"] < MIN_VEGETATION_RATIO:
        return True, "A imagem nao parece conter vegetacao/lavoura suficiente para diagnostico de soja."
    return False, None


def is_low_quality(quality: dict) -> tuple[bool, str | None]:
    if quality["sharpness"] < MIN_SHARPNESS:
        return True, "Imagem com pouca nitidez. Envie uma foto mais clara da planta/lavoura."
    return False, None


def preprocess_image(rgb_image: np.ndarray) -> tf.Tensor:
    img = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return tf.constant(img, dtype=tf.float32)


def legacy_preprocess_image(image_bytes: bytes) -> tf.Tensor:
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
        rgb_image = read_image_rgb(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    quality = analyze_quality_and_domain(rgb_image)

    out_of_domain, domain_reason = is_out_of_domain(quality)
    if out_of_domain:
        return {
            "resultado": "Nao_e_soja",
            "status": "fora_do_dominio",
            "mensagem": domain_reason,
            "qualidade": quality,
            "top3": [],
            "probabilidades": {}
        }

    low_quality, quality_reason = is_low_quality(quality)
    if low_quality:
        return {
            "resultado": "Inconclusivo",
            "status": "baixa_qualidade",
            "mensagem": quality_reason,
            "qualidade": quality,
            "top3": [],
            "probabilidades": {}
        }

    img_tensor = preprocess_image(rgb_image)

    # Inferência passando o tensor com o nome correto da chave de entrada
    result      = infer(**{input_key: img_tensor})
    predictions = result[output_key].numpy()[0]

    sorted_idx = np.argsort(predictions)[::-1]
    index      = int(sorted_idx[0])
    second_idx = int(sorted_idx[1])
    confidence = float(predictions[index])
    second_confidence = float(predictions[second_idx])
    margin     = confidence - second_confidence
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
            "status":          "baixa_confianca",
            "mensagem":        "A imagem parece ser de soja/lavoura, mas o modelo nao teve confianca suficiente.",
            "confianca":       round(confidence * 100, 2),
            "margem":          round(margin * 100, 2),
            "nivel_confianca": confidence_level(confidence),
            "qualidade":       quality,
            "top3":            top3,
            "probabilidades":  probabilities
        }

    if margin < MARGIN_THRESHOLD:
        return {
            "resultado":       "Inconclusivo",
            "status":          "classes_proximas",
            "mensagem":        "O modelo ficou dividido entre duas classes. Envie outra imagem para confirmar.",
            "confianca":       round(confidence * 100, 2),
            "margem":          round(margin * 100, 2),
            "nivel_confianca": confidence_level(confidence),
            "qualidade":       quality,
            "top3":            top3,
            "probabilidades":  probabilities
        }

    return {
        "resultado":       disease,
        "status":          "ok",
        "confianca":       round(confidence * 100, 2),
        "margem":          round(margin * 100, 2),
        "nivel_confianca": confidence_level(confidence),
        "qualidade":       quality,
        "top3":            top3,
        "probabilidades":  probabilities
    }
