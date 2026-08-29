# Soybean Disease and Damage API

[Versão em português](README.md)

A FastAPI REST API that supports the visual screening of soybean images. It accepts a single image or a batch, checks whether the input is usable, and runs a previously trained **EfficientNetB3** model.

The service is intended for field surveys and images collected with mobile devices or drones. It is a decision-support tool, not a replacement for an agronomist's phytosanitary assessment.

## Contents

- [Features](#features)
- [Recognized classes](#recognized-classes)
- [How an image is processed](#how-an-image-is-processed)
- [Requirements](#requirements)
- [Run locally](#run-locally)
- [Run with Docker](#run-with-docker)
- [API endpoints](#api-endpoints)
- [Usage examples](#usage-examples)
- [Response and status reference](#response-and-status-reference)
- [Configuration](#configuration)
- [Image capture guidelines](#image-capture-guidelines)
- [Project structure](#project-structure)
- [Limitations and responsible use](#limitations-and-responsible-use)

## Features

- Single-image classification.
- Batch processing of up to 100 images by default.
- Per-file results, including files rejected during validation.
- Validation of file type, size, resolution, integrity, visual domain, and sharpness.
- EXIF-based image orientation correction.
- Top-3 classes, class scores, confidence, and margin between the two leading classes.
- Batch summary with the predominant condition, agreement rate, and heterogeneity detection.
- Interactive Swagger UI documentation.
- Environment-based configuration and a Docker image definition.

## Recognized classes

The model included in this repository has the following output labels:

| API label | Meaning |
|---|---|
| `Ataque_de_largata_Soja` | Visual indication of soybean caterpillar damage. |
| `Cercospora` | Visual indication consistent with Cercospora. |
| `Doenca_de_Ferrugem_Soja` | Visual indication consistent with soybean rust. |
| `Soja_Saudavel` | Image classified as healthy soybean. |

These are the model's technical labels. They are returned unchanged so consuming applications can map them to their own interface language.

## How an image is processed

Each file goes through the following steps before model inference:

1. File, MIME type, size, and resolution validation.
2. Image decoding, EXIF orientation correction, and RGB conversion. JPG, PNG, WEBP, BMP, and TIFF files are accepted.
3. Basic visual checks for vegetation presence, coherent vegetation area, and sharpness.
4. Resizing to `300 × 300` pixels and EfficientNet-compatible preprocessing.
5. Model inference and evaluation of the confidence and class margin thresholds.

Images without sufficient vegetation or crop-like visual content are marked as `fora_do_dominio` (out of domain). Blurry images are marked as `baixa_qualidade` (low quality). If an otherwise valid image does not meet the decision thresholds, the API returns `Inconclusivo` (inconclusive) instead of forcing a class.

For batch requests, every image that passes the initial filters contributes to the mean class scores. The service also measures agreement between individual top predictions and returns `Lote_heterogeneo` (heterogeneous batch) when more than one reliable condition has a relevant presence.

## Requirements

To run the service directly on your machine, you need:

- Python 3.10;
- an up-to-date `pip` installation;
- the `models/` directory, including the SavedModel and `classes.json`.

> The project uses TensorFlow 2.15. Python 3.10 is recommended to avoid compatibility issues.

## Run locally

From the project root, run the following commands in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 7860
```

With the server running, open:

- Swagger UI: <http://127.0.0.1:7860/docs>
- OpenAPI specification: <http://127.0.0.1:7860/openapi.json>
- Health check: <http://127.0.0.1:7860/health>

On Linux or macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Run with Docker

Build the image:

```bash
docker build -t soybean-disease-api .
```

Run the container and expose the API port:

```bash
docker run --rm -p 7860:7860 --name soybean-disease-api soybean-disease-api
```

The interactive documentation is then available at <http://localhost:7860/docs>.

To change runtime settings without rebuilding the image, provide environment variables:

```bash
docker run --rm -p 7860:7860 \
  -e CONFIDENCE_THRESHOLD=0.70 \
  -e CORS_ORIGINS=http://localhost:3000 \
  --name soybean-disease-api \
  soybean-disease-api
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Returns service status, version, classes, and primary routes. |
| `GET` | `/health` | Confirms that the application and model are loaded. |
| `GET` | `/model-info` | Returns the model architecture, input shape, classes, and active limits. |
| `POST` | `/predict` | Classifies one image sent in the multipart `file` field. |
| `POST` | `/predict/batch` | Classifies multiple images sent in repeated multipart `files` fields. |
| `POST` | `/predict-batch` | Backward-compatible alias for `/predict/batch`. |
| `GET` | `/docs` | Swagger UI. |

### `GET /health`

Expected response:

```json
{
  "status": "ok",
  "modelo_carregado": true,
  "quantidade_classes": 4
}
```

Use this endpoint for availability checks, load balancers, or container monitoring.

### `GET /model-info`

Returns the parameters relevant to API clients, including input dimensions, request limits, and active decision thresholds. Values reflect the running instance, including any environment-variable overrides.

### `POST /predict`

Accepts one image through the required `file` field as `multipart/form-data`.

| Field | Type | Required | Description |
|---|---|---:|---|
| `file` | file | Yes | A JPG, PNG, WEBP, BMP, or TIFF image. |

### `POST /predict/batch`

Accepts one or more images through the required, repeated `files` field as `multipart/form-data`.

| Field | Type | Required | Description |
|---|---|---:|---|
| `files` | files | Yes | One or more images. Send the same field once for each file. |

The `resultados` array preserves the file submission order. Files that cannot be processed receive their own status while the remaining files continue through the pipeline.

## Usage examples

### Single image with cURL

On Windows:

```powershell
curl.exe -X POST "http://127.0.0.1:7860/predict" `
  -F "file=@C:\photos\soybean_001.jpg"
```

On Linux or macOS:

```bash
curl -X POST "http://127.0.0.1:7860/predict" \
  -F "file=@/path/to/soybean_001.jpg"
```

Example of a successful response:

```json
{
  "arquivo": "soybean_001.jpg",
  "qualidade": {
    "vegetation_ratio": 0.4812,
    "vegetation_component_ratio": 0.4339,
    "edge_density_in_vegetation": 0.1351,
    "hue_std_in_vegetation": 12.48,
    "sharpness": 82.51,
    "brightness": 118.37
  },
  "confianca": 91.24,
  "margem": 85.82,
  "nivel_confianca": "Alta",
  "top3": [
    ["Soja_Saudavel", 91.24],
    ["Cercospora", 5.42],
    ["Doenca_de_Ferrugem_Soja", 2.13]
  ],
  "probabilidades": {
    "Soja_Saudavel": 91.24,
    "Cercospora": 5.42,
    "Doenca_de_Ferrugem_Soja": 2.13,
    "Ataque_de_largata_Soja": 1.21
  },
  "resultado": "Soja_Saudavel",
  "status": "ok"
}
```

The numeric values above are illustrative. `confianca`, `margem`, and class scores are reported as percentages. `top3` is an array of `[class, percentage]` pairs. The response field names remain in Portuguese because they are part of the API contract.

### Batch request with cURL

```powershell
curl.exe -X POST "http://127.0.0.1:7860/predict/batch" `
  -F "files=@C:\photos\soybean_001.jpg" `
  -F "files=@C:\photos\soybean_002.jpg" `
  -F "files=@C:\photos\soybean_003.jpg"
```

The response includes a temporary batch ID, the aggregate result, and a diagnosis for each file:

```json
{
  "lote_id": "e7e9b5ee-6cc4-4a6e-9f7f-3d6db62c3dc4",
  "resultado_geral": {
    "total_recebidas": 3,
    "analisadas_pelo_modelo": 3,
    "resultados_confiaveis": 3,
    "inconclusivas_ou_rejeitadas": 0,
    "taxa_aproveitamento": 100.0,
    "status_das_imagens": { "ok": 3 },
    "ocorrencias_confiaveis": [
      {
        "classe": "Soja_Saudavel",
        "imagens_confiaveis": 3,
        "percentual_das_confiaveis": 100.0
      }
    ],
    "condicao_predominante": "Soja_Saudavel",
    "confianca_media": 88.76,
    "margem_media": 67.21,
    "consenso": 100.0,
    "probabilidades_medias": {
      "Soja_Saudavel": 88.76,
      "Cercospora": 7.64,
      "Doenca_de_Ferrugem_Soja": 2.31,
      "Ataque_de_largata_Soja": 1.29
    },
    "resultado": "Soja_Saudavel",
    "status": "ok",
    "mensagem": "Condição predominante estimada a partir do conjunto de imagens."
  },
  "resultados": [],
  "tempo_processamento_ms": 412.58,
  "aviso": "Resultado de apoio à triagem. A confirmação agronômica continua necessária, especialmente em lotes heterogêneos."
}
```

The `resultados` array was shortened in this example for readability. In an actual response, it contains the full result for every submitted image.

### JavaScript integration

```javascript
async function analyzePhotos(files) {
  const formData = new FormData();

  for (const file of files) {
    formData.append("files", file);
  }

  const response = await fetch("http://127.0.0.1:7860/predict/batch", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || "Unable to analyze the images.");
  }

  return response.json();
}
```

Do not set the `Content-Type` header manually when using `FormData`; the browser adds the required multipart boundary automatically.

## Response and status reference

### Per-image statuses

| Status | Result | Meaning |
|---|---|---|
| `ok` | Class name | The prediction meets both confidence and margin thresholds. |
| `baixa_confianca` | `Inconclusivo` | The leading score did not reach the minimum confidence threshold. |
| `classes_proximas` | `Inconclusivo` | The top two classes are too close to make a reliable decision. |
| `fora_do_dominio` | `Nao_e_soja` | There is not enough visual evidence of compatible vegetation/cropland. |
| `baixa_qualidade` | `Inconclusivo` | The image does not meet the minimum sharpness requirement. |
| `arquivo_vazio` | `Inconclusivo` | The submitted file is empty. |
| `arquivo_muito_grande` | `Inconclusivo` | The file exceeds the configured individual size limit. |
| `tipo_invalido` | `Inconclusivo` | The submitted MIME type is not accepted. |
| `imagem_invalida` | `Inconclusivo` | The content could not be decoded as a valid image. |

For the single-image endpoint, file errors return HTTP `400`, except `arquivo_muito_grande`, which returns HTTP `413`. In batch processing, these conditions are recorded in the corresponding item; the request only fails when the entire batch violates a rule, such as no files being supplied, too many files, or exceeding the total request size.

### Batch statuses

| Status | Result | Meaning |
|---|---|---|
| `ok` | Class name | Confidence, margin, and agreement are sufficient for a predominant condition. |
| `heterogeneo` | `Lote_heterogeneo` | More than one reliable condition has a relevant presence. |
| `consenso_insuficiente` | `Inconclusivo` | The batch lacks sufficient confidence, margin, or agreement. |
| `sem_imagens_analisaveis` | `Inconclusivo` | No image passed the initial validation checks. |

`consenso` is the percentage of model-analyzed images whose top-1 class matches the predominant condition. `ocorrencias_confiaveis` only counts individual results with status `ok`.

## Configuration

Use the following environment variables to adjust the service without changing source code.

| Variable | Default | Description |
|---|---:|---|
| `MODEL_PATH` | `models/modelo_ml_savedmodel` | Directory containing the TensorFlow SavedModel. |
| `CLASSES_PATH` | `models/classes.json` | JSON file with the model output labels. |
| `IMG_SIZE` | `300` | Model input width and height in pixels. |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum confidence for accepting an individual or aggregated prediction. |
| `MARGIN_THRESHOLD` | `0.18` | Minimum difference between the two leading classes. |
| `BATCH_SUPPORT_THRESHOLD` | `0.50` | Minimum agreement required to accept the batch's predominant result. |
| `MAX_IMAGE_SIZE_MB` | `20` | Maximum allowed size for each image. |
| `MAX_BATCH_SIZE_MB` | `500` | Maximum combined size for files in one batch. |
| `MAX_BATCH_FILES` | `100` | Maximum image count in one batch. |
| `MODEL_BATCH_SIZE` | `16` | Number of images passed to the model in each inference block. |
| `MAX_IMAGE_PIXELS` | `50000000` | Maximum pixel count for a decoded image. |
| `QUALITY_MAX_EDGE` | `1600` | Maximum image edge used during quality and domain checks. |
| `CORS_ORIGINS` | `*` | Allowed origins, separated by commas. |

PowerShell example:

```powershell
$env:CONFIDENCE_THRESHOLD = "0.70"
$env:CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:5173"
uvicorn main:app --host 0.0.0.0 --port 7860
```

> For production, set `CORS_ORIGINS` to the exact frontend domains. Avoid leaving `*` in place when the API is publicly exposed.

## Image capture guidelines

Image quality has a direct effect on the usefulness of the result. For more consistent screening:

- keep the plant or canopy in focus and properly lit;
- avoid images taken from excessive altitude, images with blur, strong backlight, or heavy compression;
- capture different areas of the field instead of making a decision from one photo;
- for drone surveys, use a flight height and resolution that preserve relevant visual details;
- review individual results whenever a batch is heterogeneous or inconclusive;
- validate every suspected disease or pest in the field before defining a management action.

## Project structure

```text
.
├── main.py                       # FastAPI application, inference, and routes
├── domain_gate.py                # Visual domain and image quality criteria
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Application container definition
└── models/
    ├── classes.json              # Labels associated with model outputs
    └── modelo_ml_savedmodel/     # TensorFlow SavedModel
```

## Limitations and responsible use

- This API classifies images; it does not provide a definitive agronomic diagnosis.
- Performance depends on the representativeness of training data and on image-capture conditions. The repository does not include model validation metrics, so accuracy or performance under unevaluated conditions should not be assumed.
- The domain check is a simple visual filter intended to reject clearly unsuitable inputs. It does not replace agronomic validation or guarantee that every accepted image is soybean.
- High-altitude aerial images may lose lesion detail when resized to the model input dimensions.
- Treat `Inconclusivo` and `Lote_heterogeneo` as signals to review the images and inspect the field.

For academic or operational use, record the capture protocol, equipment, flight altitude where applicable, sampled area, and specialist validation. This context is essential for responsible interpretation of the results.
