---
title: API de Doenças e Danos na Soja
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
sdk_version: "0.0.1"
python_version: "3.10"
app_file: main.py
pinned: false
---

# API de doenças e danos na soja

API FastAPI que usa o modelo EfficientNetB3 já treinado para analisar uma imagem
ou um lote de imagens coletadas por drone. A análise em lote não exige novo
treinamento: o mesmo SavedModel é executado com várias imagens por inferência.

## Endpoints

| Método | Rota | Finalidade |
|---|---|---|
| `GET` | `/health` | Verifica se a API e o modelo estão disponíveis |
| `GET` | `/model-info` | Mostra classes, limites e tamanho de entrada |
| `POST` | `/predict` | Analisa uma imagem no campo multipart `file` |
| `POST` | `/predict/batch` | Analisa até 100 imagens usando o campo multipart `files` |
| `GET` | `/docs` | Interface Swagger para testar a API |

`/predict-batch` continua disponível como alias de `/predict/batch`.

## Execução local

Requer Python 3.10, porque o projeto utiliza TensorFlow 2.15.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 7860
```

Abra `http://127.0.0.1:7860/docs` para enviar uma ou várias fotos sem criar um
front-end.

## Exemplos

Uma imagem:

```powershell
curl.exe -X POST "http://127.0.0.1:7860/predict" `
  -F "file=@C:\fotos\soja_001.jpg"
```

Várias imagens no mesmo lote:

```powershell
curl.exe -X POST "http://127.0.0.1:7860/predict/batch" `
  -F "files=@C:\fotos\soja_001.jpg" `
  -F "files=@C:\fotos\soja_002.jpg" `
  -F "files=@C:\fotos\soja_003.jpg"
```

Exemplo JavaScript para um seletor `<input type="file" multiple>`:

```javascript
async function analisarFotos(arquivos) {
  const formData = new FormData();

  for (const arquivo of arquivos) {
    formData.append("files", arquivo);
  }

  const resposta = await fetch("http://127.0.0.1:7860/predict/batch", {
    method: "POST",
    body: formData,
  });

  if (!resposta.ok) {
    throw new Error(await resposta.text());
  }

  return resposta.json();
}
```

## Como interpretar o lote

A resposta de `/predict/batch` contém:

- `resultado_geral`: consolidação do lote;
- `condicao_predominante`: classe com maior probabilidade média;
- `consenso`: percentual das imagens analisadas cujo Top-1 concorda com a
  condição predominante;
- `ocorrencias_confiaveis`: distribuição das imagens individuais que passaram
  pelos limiares de confiança;
- `resultados`: diagnóstico e controle de qualidade de cada arquivo;
- `heterogeneo`: indica que mais de uma condição apareceu com presença
  relevante. Isso não deve ser forçado para uma única classe.

A média considera todas as imagens que passaram pelos filtros de domínio e
qualidade. As ocorrências usam somente resultados individuais classificados como
`ok`.

## Limites configuráveis

As variáveis de ambiente abaixo permitem ajustar o serviço sem mudar o código:

| Variável | Padrão | Descrição |
|---|---:|---|
| `MAX_BATCH_FILES` | `100` | Quantidade máxima de imagens por requisição |
| `MAX_IMAGE_SIZE_MB` | `20` | Limite por imagem |
| `MAX_BATCH_SIZE_MB` | `500` | Limite total da requisição |
| `MODEL_BATCH_SIZE` | `16` | Imagens processadas juntas pelo modelo |
| `CONFIDENCE_THRESHOLD` | `0.65` | Confiança mínima individual e agregada |
| `MARGIN_THRESHOLD` | `0.18` | Diferença mínima entre as duas primeiras classes |
| `BATCH_SUPPORT_THRESHOLD` | `0.50` | Consenso mínimo para o resultado do lote |
| `CORS_ORIGINS` | `*` | Origens permitidas, separadas por vírgula |

Em levantamentos maiores, o cliente deve enviar blocos de até 100 imagens. Para
milhares de fotos, o passo seguinte é uma fila assíncrona com armazenamento e
identificador persistente de missão.

## Limitação científica

O resultado é apoio à triagem, não confirmação fitossanitária. Imagens aéreas
muito altas podem não conservar detalhes de lesões após o redimensionamento para
300 × 300. O protocolo de voo, a altura, a câmera e a validação por profissional
da área devem constar no TCC.
