import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input

from domain_gate import analyze_quality_and_domain, is_low_quality, is_out_of_domain


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "modelo_ml_savedmodel"))
)
CLASSES_PATH = Path(
    os.getenv("CLASSES_PATH", str(BASE_DIR / "models" / "classes.json"))
)

IMG_SIZE = int(os.getenv("IMG_SIZE", "300"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.18"))
BATCH_SUPPORT_THRESHOLD = float(os.getenv("BATCH_SUPPORT_THRESHOLD", "0.50"))

MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))
MAX_BATCH_SIZE_MB = int(os.getenv("MAX_BATCH_SIZE_MB", "500"))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "100"))
MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", "16"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
QUALITY_MAX_EDGE = int(os.getenv("QUALITY_MAX_EDGE", "1600"))

IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Imagens truncadas devem ser recusadas em uma API de diagnóstico.
ImageFile.LOAD_TRUNCATED_IMAGES = False


def _load_runtime() -> tuple[Any, Any, str, str, list[str]]:
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"Modelo não encontrado em: {MODEL_PATH}")
    if not CLASSES_PATH.is_file():
        raise RuntimeError(f"Arquivo de classes não encontrado em: {CLASSES_PATH}")

    loaded_model = tf.saved_model.load(str(MODEL_PATH))
    try:
        signature = loaded_model.signatures["serving_default"]
    except KeyError as exc:
        raise RuntimeError("O SavedModel não possui a assinatura 'serving_default'.") from exc

    signature_inputs = signature.structured_input_signature[1]
    signature_outputs = signature.structured_outputs
    if len(signature_inputs) != 1 or len(signature_outputs) != 1:
        raise RuntimeError("A API espera um modelo com uma entrada e uma saída.")

    with CLASSES_PATH.open("r", encoding="utf-8") as classes_file:
        loaded_classes = json.load(classes_file)

    if not isinstance(loaded_classes, list) or not loaded_classes:
        raise RuntimeError("classes.json deve conter uma lista não vazia de classes.")

    return (
        loaded_model,
        signature,
        next(iter(signature_inputs)),
        next(iter(signature_outputs)),
        loaded_classes,
    )


print("Carregando modelo de soja...")
model, infer, input_key, output_key, classes = _load_runtime()
print(f"Modelo carregado. Entrada='{input_key}', saída='{output_key}', classes={classes}")

# Evita duas inferências simultâneas disputando a mesma memória de CPU/GPU.
inference_lock = threading.Lock()


cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(
    title="API - Detecção de Doenças e Danos na Soja",
    description=(
        "Classifica uma foto ou um lote de fotos de soja com EfficientNetB3 "
        "e consolida os resultados de levantamentos por drone."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def safe_filename(filename: str | None, position: int) -> str:
    name = Path(filename or f"imagem_{position + 1}").name
    return name or f"imagem_{position + 1}"


def read_image_rgb(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Resolução inválida ou acima do limite de {MAX_IMAGE_PIXELS:,} pixels."
                )
            image = ImageOps.exif_transpose(image)
            return np.asarray(image.convert("RGB"))
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        np_image = np.frombuffer(image_bytes, np.uint8)
        decoded = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Imagem inválida ou corrompida.")

        height, width = decoded.shape[:2]
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Resolução acima do limite de {MAX_IMAGE_PIXELS:,} pixels."
            )
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def resize_long_edge(rgb_image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_edge:
        return rgb_image

    scale = max_edge / float(current_max)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(
        rgb_image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_image(rgb_image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        rgb_image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    prepared = preprocess_input(resized.astype(np.float32))
    return np.asarray(prepared, dtype=np.float32)


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "Alta"
    if confidence >= 0.60:
        return "Média"
    return "Baixa"


def empty_prediction(filename: str, status: str, message: str) -> dict[str, Any]:
    return {
        "arquivo": filename,
        "resultado": "Inconclusivo",
        "status": status,
        "mensagem": message,
        "top3": [],
        "probabilidades": {},
    }


def prepare_image(
    filename: str,
    content_type: str | None,
    contents: bytes,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    if not contents:
        return empty_prediction(filename, "arquivo_vazio", "O arquivo está vazio."), None

    if len(contents) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        return (
            empty_prediction(
                filename,
                "arquivo_muito_grande",
                f"A imagem excede o limite de {MAX_IMAGE_SIZE_MB} MB.",
            ),
            None,
        )

    normalized_content_type = (content_type or "").split(";", maxsplit=1)[0].lower()
    if normalized_content_type and normalized_content_type not in IMAGE_CONTENT_TYPES:
        return (
            empty_prediction(
                filename,
                "tipo_invalido",
                "Envie uma imagem JPG, PNG, WEBP, BMP ou TIFF.",
            ),
            None,
        )

    try:
        rgb_image = read_image_rgb(contents)
    except ValueError as exc:
        return empty_prediction(filename, "imagem_invalida", str(exc)), None

    quality_image = resize_long_edge(rgb_image, QUALITY_MAX_EDGE)
    quality = analyze_quality_and_domain(quality_image)

    out_of_domain, domain_reason = is_out_of_domain(quality)
    if out_of_domain:
        return (
            {
                "arquivo": filename,
                "resultado": "Nao_e_soja",
                "status": "fora_do_dominio",
                "mensagem": domain_reason,
                "qualidade": quality,
                "top3": [],
                "probabilidades": {},
            },
            None,
        )

    low_quality, quality_reason = is_low_quality(quality)
    if low_quality:
        return (
            {
                "arquivo": filename,
                "resultado": "Inconclusivo",
                "status": "baixa_qualidade",
                "mensagem": quality_reason,
                "qualidade": quality,
                "top3": [],
                "probabilidades": {},
            },
            None,
        )

    return {"arquivo": filename, "qualidade": quality}, preprocess_image(rgb_image)


def run_model_batch(prepared_images: list[np.ndarray]) -> np.ndarray:
    if not prepared_images:
        return np.empty((0, len(classes)), dtype=np.float32)

    prediction_chunks: list[np.ndarray] = []
    with inference_lock:
        for start in range(0, len(prepared_images), MODEL_BATCH_SIZE):
            image_chunk = np.stack(
                prepared_images[start : start + MODEL_BATCH_SIZE],
                axis=0,
            )
            output = infer(**{input_key: tf.constant(image_chunk, dtype=tf.float32)})
            prediction_chunks.append(np.asarray(output[output_key].numpy()))

    predictions = np.concatenate(prediction_chunks, axis=0)
    if predictions.ndim != 2 or predictions.shape[1] != len(classes):
        raise RuntimeError(
            "A quantidade de saídas do modelo não corresponde ao arquivo classes.json."
        )
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("O modelo retornou probabilidades inválidas.")
    return predictions


def format_prediction(
    base_result: dict[str, Any],
    prediction: np.ndarray,
) -> dict[str, Any]:
    sorted_indices = np.argsort(prediction)[::-1]
    best_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1])
    confidence = float(prediction[best_index])
    second_confidence = float(prediction[second_index])
    margin = confidence - second_confidence

    probabilities = {
        classes[index]: round(float(prediction[index]) * 100, 2)
        for index in sorted_indices
    }
    common = {
        **base_result,
        "confianca": round(confidence * 100, 2),
        "margem": round(margin * 100, 2),
        "nivel_confianca": confidence_level(confidence),
        "top3": list(probabilities.items())[:3],
        "probabilidades": probabilities,
    }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            **common,
            "resultado": "Inconclusivo",
            "status": "baixa_confianca",
            "mensagem": (
                "A imagem passou pelo controle de qualidade, mas o modelo não "
                "atingiu a confiança mínima."
            ),
        }

    if margin < MARGIN_THRESHOLD:
        return {
            **common,
            "resultado": "Inconclusivo",
            "status": "classes_proximas",
            "mensagem": "O modelo ficou dividido entre as duas classes mais prováveis.",
        }

    return {
        **common,
        "resultado": classes[best_index],
        "status": "ok",
    }


async def analyze_uploads(
    files: list[UploadFile],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    if not files:
        raise HTTPException(status_code=400, detail="Envie ao menos uma imagem.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo de {MAX_BATCH_FILES} imagens por lote.",
        )

    results: list[dict[str, Any] | None] = [None] * len(files)
    prepared_images: list[np.ndarray] = []
    prepared_positions: list[int] = []
    total_bytes = 0

    for position, upload in enumerate(files):
        filename = safe_filename(upload.filename, position)
        content_type = upload.content_type
        try:
            contents = await upload.read()
        finally:
            await upload.close()

        total_bytes += len(contents)
        if total_bytes > MAX_BATCH_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"O lote excede o limite total de {MAX_BATCH_SIZE_MB} MB.",
            )

        preliminary_result, prepared = prepare_image(
            filename,
            content_type,
            contents,
        )
        if prepared is None:
            results[position] = preliminary_result
            continue

        results[position] = preliminary_result
        prepared_positions.append(position)
        prepared_images.append(prepared)

    predictions_by_position: dict[int, np.ndarray] = {}
    if prepared_images:
        try:
            predictions = run_model_batch(prepared_images)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha durante a inferência do modelo: {exc}",
            ) from exc

        for position, prediction in zip(prepared_positions, predictions, strict=True):
            base_result = results[position]
            if base_result is None:
                raise RuntimeError("Estado interno inválido durante a predição.")
            results[position] = format_prediction(base_result, prediction)
            predictions_by_position[position] = prediction

    if any(result is None for result in results):
        raise RuntimeError("Nem todas as imagens receberam um resultado.")

    return [result for result in results if result is not None], predictions_by_position


def aggregate_batch(
    results: list[dict[str, Any]],
    predictions_by_position: dict[int, np.ndarray],
) -> dict[str, Any]:
    total = len(results)
    modeled_positions = sorted(predictions_by_position)
    modeled_count = len(modeled_positions)
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    reliable_results = [result for result in results if result["status"] == "ok"]
    reliable_counts = {class_name: 0 for class_name in classes}
    for result in reliable_results:
        reliable_counts[str(result["resultado"])] += 1

    occurrences = [
        {
            "classe": class_name,
            "imagens_confiaveis": count,
            "percentual_das_confiaveis": round(
                (count / len(reliable_results) * 100) if reliable_results else 0.0,
                2,
            ),
        }
        for class_name, count in sorted(
            reliable_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if count > 0
    ]

    summary: dict[str, Any] = {
        "total_recebidas": total,
        "analisadas_pelo_modelo": modeled_count,
        "resultados_confiaveis": len(reliable_results),
        "inconclusivas_ou_rejeitadas": total - len(reliable_results),
        "taxa_aproveitamento": round(modeled_count / total * 100, 2) if total else 0.0,
        "status_das_imagens": status_counts,
        "ocorrencias_confiaveis": occurrences,
    }

    if modeled_count == 0:
        return {
            **summary,
            "resultado": "Inconclusivo",
            "status": "sem_imagens_analisaveis",
            "mensagem": "Nenhuma imagem do lote passou pelos controles de domínio e qualidade.",
            "condicao_predominante": None,
            "consenso": 0.0,
            "probabilidades_medias": {},
        }

    prediction_matrix = np.stack(
        [predictions_by_position[position] for position in modeled_positions],
        axis=0,
    )
    mean_prediction = prediction_matrix.mean(axis=0)
    sorted_indices = np.argsort(mean_prediction)[::-1]
    best_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1])
    batch_confidence = float(mean_prediction[best_index])
    batch_margin = batch_confidence - float(mean_prediction[second_index])

    top_indices = np.argmax(prediction_matrix, axis=1)
    support = float(np.mean(top_indices == best_index))
    mean_probabilities = {
        classes[index]: round(float(mean_prediction[index]) * 100, 2)
        for index in sorted_indices
    }

    common = {
        **summary,
        "condicao_predominante": classes[best_index],
        "confianca_media": round(batch_confidence * 100, 2),
        "margem_media": round(batch_margin * 100, 2),
        "consenso": round(support * 100, 2),
        "probabilidades_medias": mean_probabilities,
    }

    substantial_conditions = [
        item
        for item in occurrences
        if item["percentual_das_confiaveis"] >= 20.0
    ]
    if len(substantial_conditions) > 1:
        return {
            **common,
            "resultado": "Lote_heterogeneo",
            "status": "heterogeneo",
            "mensagem": (
                "O lote contém mais de uma condição com presença relevante. "
                "Consulte os resultados por imagem para localizar as áreas."
            ),
        }

    if (
        batch_confidence < CONFIDENCE_THRESHOLD
        or batch_margin < MARGIN_THRESHOLD
        or support < BATCH_SUPPORT_THRESHOLD
    ):
        return {
            **common,
            "resultado": "Inconclusivo",
            "status": "consenso_insuficiente",
            "mensagem": (
                "As imagens não produziram confiança e consenso suficientes "
                "para um resultado predominante do lote."
            ),
        }

    return {
        **common,
        "resultado": classes[best_index],
        "status": "ok",
        "mensagem": "Condição predominante estimada a partir do conjunto de imagens.",
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "online",
        "versao": app.version,
        "modelo": "EfficientNetB3",
        "classes": classes,
        "endpoints": {
            "uma_imagem": "POST /predict (campo: file)",
            "varias_imagens": "POST /predict/batch (campo repetido: files)",
            "documentacao": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "modelo_carregado": True,
        "quantidade_classes": len(classes),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "arquitetura": "EfficientNetB3",
        "tamanho_entrada": [IMG_SIZE, IMG_SIZE, 3],
        "classes": classes,
        "limiar_confianca": CONFIDENCE_THRESHOLD,
        "limiar_margem": MARGIN_THRESHOLD,
        "maximo_imagens_por_lote": MAX_BATCH_FILES,
        "maximo_mb_por_imagem": MAX_IMAGE_SIZE_MB,
        "tamanho_bloco_inferencia": MODEL_BATCH_SIZE,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    results, _ = await analyze_uploads([file])
    result = results[0]

    client_error_statuses = {
        "arquivo_vazio",
        "arquivo_muito_grande",
        "tipo_invalido",
        "imagem_invalida",
    }
    if result["status"] in client_error_statuses:
        status_code = 413 if result["status"] == "arquivo_muito_grande" else 400
        raise HTTPException(status_code=status_code, detail=result["mensagem"])
    return result


@app.post("/predict-batch", include_in_schema=False)
@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    started_at = time.perf_counter()
    results, predictions_by_position = await analyze_uploads(files)
    aggregate = aggregate_batch(results, predictions_by_position)

    return {
        "lote_id": str(uuid4()),
        "resultado_geral": aggregate,
        "resultados": results,
        "tempo_processamento_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "aviso": (
            "Resultado de apoio à triagem. A confirmação agronômica continua "
            "necessária, especialmente em lotes heterogêneos."
        ),
    }
