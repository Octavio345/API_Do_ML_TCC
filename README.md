# 🌱 API — Detecção de Doenças na Soja · Soybean Disease Detection API

> 🇧🇷 [Português](#português) · 🇺🇸 [English](#english)

---

<a name="português"></a>
# 🇧🇷 Português

# 🌱 API — Detecção de Doenças na Soja

> Classificação automática de doenças em plantações de soja a partir de imagens, utilizando **EfficientNetB3** com TensorFlow.

---

## Índice

- [Visão Geral](#visão-geral)
- [Stack Técnica](#stack-técnica)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [Instalação e Execução](#instalação-e-execução)
- [Configurações](#configurações)
- [Endpoints](#endpoints)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [POST /predict](#post-predict)
- [Exemplos de Integração](#exemplos-de-integração)
- [Códigos de Erro](#códigos-de-erro)
- [Nível de Confiança](#nível-de-confiança)

---

## Visão Geral

API REST assíncrona construída com **FastAPI** para classificação de doenças em lavouras de soja. O modelo recebe uma imagem da plantação e retorna a doença detectada, o percentual de confiança e as probabilidades de todas as classes.

---

## Stack Técnica

| Componente | Tecnologia |
|---|---|
| Framework web | FastAPI |
| Modelo ML | EfficientNetB3 (TensorFlow SavedModel) |
| Processamento de imagem | OpenCV (cv2) |
| Operações matriciais | NumPy |
| Inferência | TensorFlow 2.x |

---

## Estrutura de Arquivos

```
projeto/
├── main.py
└── models/
    ├── modelo_ml_savedmodel/      # SavedModel exportado pelo Keras/TF
    │   ├── saved_model.pb
    │   └── variables/
    └── classes.json               # Lista ordenada de rótulos de classe
```

**Exemplo de `classes.json`:**
```json
["Ferrugem Asiática", "Cercospora", "Ataque de Largata", "Saudável"],
```

---

## Instalação e Execução

**1. Instale as dependências:**
```bash
pip install fastapi uvicorn tensorflow opencv-python numpy
```

**2. Inicie o servidor:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Acesse a documentação interativa:**
- Swagger UI → http://localhost:8000/docs
- ReDoc → http://localhost:8000/redoc

---

## Configurações

As constantes abaixo controlam o comportamento da API e podem ser ajustadas diretamente em `main.py`:

| Variável | Padrão | Descrição |
|---|---|---|
| `IMG_SIZE` | `300` | Dimensão de redimensionamento da imagem (px) |
| `CONFIDENCE_THRESHOLD` | `0.50` | Limiar mínimo de confiança para resultado definitivo |
| `MAX_IMAGE_SIZE_MB` | `5` | Tamanho máximo do upload em MB |
| `MODEL_PATH` | `"models/modelo_ml_savedmodel"` | Caminho para o SavedModel |
| `CLASSES_PATH` | `"models/classes.json"` | Caminho para o JSON de classes |

---

## Endpoints

### GET /

Retorna informações gerais sobre a API: status, modelo carregado e classes disponíveis.

**Resposta `200 OK`:**
```json
{
  "status": "online",
  "modelo": "EfficientNetB3",
  "classes": ["Ferrugem Asiática", "Cercospora", "Ataque de Largata", "Saudável"],
  "uso": "POST /predict com uma imagem"
}
```

---

### GET /health

Health-check leve para monitoramento e orquestradores (Kubernetes, Docker, etc.).

**Resposta `200 OK`:**
```json
{"status": "ok"}
```

---

### POST /predict

Recebe uma imagem de lavoura e retorna a classificação da doença detectada.

**Requisição:**

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Arquivo de imagem (jpg, png, webp, bmp) |

- **Content-Type:** `multipart/form-data`
- **Tamanho máximo:** 5 MB
- **Formatos aceitos:** JPEG, PNG, BMP, WEBP (qualquer formato suportado pelo OpenCV)

**Pré-processamento aplicado internamente:**
1. Decodificação via OpenCV (BGR → RGB)
2. Redimensionamento para 300×300 px
3. Normalização via `EfficientNet preprocess_input`
4. Expansão de dimensão de batch — shape final: `(1, 300, 300, 3)`

---

**Resposta `200 OK` — confiança ≥ 50%:**
```json
{
  "resultado": "Ferrugem Asiática",
  "confianca": 87.34,
  "nivel_confianca": "Alta",
  "top3": [
    ["Ferrugem Asiática", 87.34],
    ["Mancha Alvo", 8.12],
    ["Saudável", 3.05]
  ],
  "probabilidades": {
    "Ferrugem Asiática": 87.34,
    "Mancha Alvo": 8.12,
    "Saudável": 3.05
  }
}
```

**Resposta `200 OK` — confiança < 50% (inconclusivo):**
```json
{
  "resultado": "Inconclusivo",
  "confianca": 41.20,
  "nivel_confianca": "Baixa",
  "top3": [...],
  "probabilidades": {...}
}
```

> Mesmo abaixo do limiar, `top3` e `probabilidades` são retornados para análise auxiliar.

---

## Exemplos de Integração

**Chamada simples com `requests`:**
```python
import requests

with open("foto_folha.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("foto_folha.jpg", f, "image/jpeg")}
    )

result = r.json()
print(result["resultado"])        # "Ferrugem Asiática"
print(result["confianca"])        # 87.34
print(result["nivel_confianca"])  # "Alta"
```

**Verificar se a API está online:**
```python
r = requests.get("http://localhost:8000/health")
if r.json()["status"] == "ok":
    print("API online!")
```

**Consultar classes disponíveis:**
```python
r = requests.get("http://localhost:8000/")
classes = r.json()["classes"]
print(classes)
```

**Processar um diretório inteiro de imagens:**
```python
from pathlib import Path

def batch_predict(folder: str, base_url: str = "http://localhost:8000") -> list:
    results = []
    for img_path in Path(folder).glob("*.jpg"):
        with open(img_path, "rb") as f:
            r = requests.post(f"{base_url}/predict", files={"file": f})
        data = r.json()
        results.append({
            "arquivo":  img_path.name,
            "resultado": data["resultado"],
            "confianca": data["confianca"],
        })
    return results
```

---

## Códigos de Erro

| Código | Motivo |
|---|---|
| `400` | Imagem maior que 5 MB |
| `400` | `Content-Type` não é imagem |
| `400` | Arquivo corrompido ou não decodificável |
| `422` | Campo `file` ausente no formulário (validação do FastAPI) |
| `500` | Erro interno não tratado |

**Exemplo de resposta de erro:**
```json
{"detail": "Imagem muito grande. Máximo: 5MB"}
```

---

## Nível de Confiança

| Confiança | Nível | Recomendação |
|---|---|---|
| > 80% | **Alta** | Resultado confiável — ação pode ser tomada |
| 60% – 80% | **Média** | Recomenda-se confirmação visual complementar |
| < 60% | **Baixa** | Resultado incerto — análise presencial indicada |
| < 50% | **Baixa*** | Retornado como `"Inconclusivo"` |

---

<a name="english"></a>
# 🇺🇸 English

# 🌱 API — Soybean Disease Detection

> Automatic classification of diseases in soybean crops from images, using **EfficientNetB3** with TensorFlow.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [File Structure](#file-structure)
- [Installation & Running](#installation--running)
- [Configuration](#configuration)
- [Endpoints](#endpoints-1)
  - [GET /](#get--1)
  - [GET /health](#get-health-1)
  - [POST /predict](#post-predict-1)
- [Integration Examples](#integration-examples)
- [Error Codes](#error-codes)
- [Confidence Level](#confidence-level)

---

## Overview

Asynchronous REST API built with **FastAPI** for classifying diseases in soybean crops. The model receives a crop image and returns the detected disease, confidence percentage, and probabilities for all classes.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI |
| ML model | EfficientNetB3 (TensorFlow SavedModel) |
| Image processing | OpenCV (cv2) |
| Matrix operations | NumPy |
| Inference | TensorFlow 2.x |

---

## File Structure

```
project/
├── main.py
└── models/
    ├── modelo_ml_savedmodel/      # Keras/TF exported SavedModel
    │   ├── saved_model.pb
    │   └── variables/
    └── classes.json               # Ordered list of class labels
```

**Example `classes.json`:**
```json
["Asian Rust", "Cercospora", "Insect Damage", "Healthy"]
```

---

## Installation & Running

**1. Install dependencies:**
```bash
pip install fastapi uvicorn tensorflow opencv-python numpy
```

**2. Start the server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Access the interactive documentation:**
- Swagger UI → http://localhost:8000/docs
- ReDoc → http://localhost:8000/redoc

---

## Configuration

The constants below control the API behaviour and can be adjusted directly in `main.py`:

| Variable | Default | Description |
|---|---|---|
| `IMG_SIZE` | `300` | Image resize dimension (px) |
| `CONFIDENCE_THRESHOLD` | `0.50` | Minimum confidence threshold for a definitive result |
| `MAX_IMAGE_SIZE_MB` | `5` | Maximum upload size in MB |
| `MODEL_PATH` | `"models/modelo_ml_savedmodel"` | Path to the SavedModel |
| `CLASSES_PATH` | `"models/classes.json"` | Path to the classes JSON |

---

## Endpoints

### GET /

Returns general API information: status, loaded model, and available classes.

**Response `200 OK`:**
```json
{
  "status": "online",
  "modelo": "EfficientNetB3",
  "classes": ["Asian Rust", "Cercospora", "Insect Damage", "Healthy"],
  "uso": "POST /predict with an image"
}
```

---

### GET /health

Lightweight health-check for monitoring and orchestrators (Kubernetes, Docker, etc.).

**Response `200 OK`:**
```json
{"status": "ok"}
```

---

### POST /predict

Receives a crop image and returns the classification of the detected disease.

**Request:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Image file (jpg, png, webp, bmp) |

- **Content-Type:** `multipart/form-data`
- **Maximum size:** 5 MB
- **Accepted formats:** JPEG, PNG, BMP, WEBP (any format supported by OpenCV)

**Pre-processing applied internally:**
1. Decoding via OpenCV (BGR → RGB)
2. Resizing to 300×300 px
3. Normalisation via `EfficientNet preprocess_input`
4. Batch dimension expansion — final shape: `(1, 300, 300, 3)`

---

**Response `200 OK` — confidence ≥ 50%:**
```json
{
  "resultado": "Asian Rust",
  "confianca": 87.34,
  "nivel_confianca": "High",
  "top3": [
    ["Asian Rust", 87.34],
    ["Target Spot", 8.12],
    ["Healthy", 3.05]
  ],
  "probabilidades": {
    "Asian Rust": 87.34,
    "Target Spot": 8.12,
    "Healthy": 3.05
  }
}
```

**Response `200 OK` — confidence < 50% (inconclusive):**
```json
{
  "resultado": "Inconclusivo",
  "confianca": 41.20,
  "nivel_confianca": "Low",
  "top3": [...],
  "probabilidades": {...}
}
```

> Even below the threshold, `top3` and `probabilidades` are returned for auxiliary analysis.

---

## Integration Examples

**Simple call with `requests`:**
```python
import requests

with open("leaf_photo.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("leaf_photo.jpg", f, "image/jpeg")}
    )

result = r.json()
print(result["resultado"])        # "Asian Rust"
print(result["confianca"])        # 87.34
print(result["nivel_confianca"])  # "High"
```

**Check if the API is online:**
```python
r = requests.get("http://localhost:8000/health")
if r.json()["status"] == "ok":
    print("API is online!")
```

**Fetch available classes:**
```python
r = requests.get("http://localhost:8000/")
classes = r.json()["classes"]
print(classes)
```

**Process an entire folder of images:**
```python
from pathlib import Path

def batch_predict(folder: str, base_url: str = "http://localhost:8000") -> list:
    results = []
    for img_path in Path(folder).glob("*.jpg"):
        with open(img_path, "rb") as f:
            r = requests.post(f"{base_url}/predict", files={"file": f})
        data = r.json()
        results.append({
            "file":       img_path.name,
            "result":     data["resultado"],
            "confidence": data["confianca"],
        })
    return results
```

---

## Error Codes

| Code | Reason |
|---|---|
| `400` | Image larger than 5 MB |
| `400` | `Content-Type` is not an image |
| `400` | Corrupted or undecodable file |
| `422` | `file` field missing from form (FastAPI validation) |
| `500` | Unhandled internal error |

**Example error response:**
```json
{"detail": "Imagem muito grande. Máximo: 5MB"}
```

---

## Confidence Level

| Confidence | Level | Recommendation |
|---|---|---|
| > 80% | **High** | Reliable result — action can be taken |
| 60% – 80% | **Medium** | Visual confirmation recommended |
| < 60% | **Low** | Uncertain result — in-person analysis advised |
| < 50% | **Low*** | Returned as `"Inconclusivo"` |
