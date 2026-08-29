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

# API de Doenças e Danos na Soja

[English version](README.en.md)

API REST desenvolvida em FastAPI para apoiar a triagem visual de imagens de soja. O serviço recebe uma imagem isolada ou um conjunto de imagens, aplica verificações de domínio e qualidade e executa a classificação com um modelo **EfficientNetB3** previamente treinado.

O objetivo é facilitar levantamentos de campo e análises de imagens captadas por dispositivos móveis ou drones. A resposta da API deve ser usada como apoio à tomada de decisão; a confirmação fitossanitária continua sendo responsabilidade de um profissional habilitado.

## Índice

- [Recursos](#recursos)
- [Classes reconhecidas](#classes-reconhecidas)
- [Como a análise funciona](#como-a-análise-funciona)
- [Pré-requisitos](#pré-requisitos)
- [Execução local](#execução-local)
- [Teste online opcional](#teste-online-opcional)
- [Execução com Docker](#execução-com-docker)
- [Rotas da API](#rotas-da-api)
- [Exemplos de uso](#exemplos-de-uso)
- [Respostas e status](#respostas-e-status)
- [Configuração](#configuração)
- [Boas práticas para captura](#boas-práticas-para-captura)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Limitações e uso responsável](#limitações-e-uso-responsável)

## Recursos

- Classificação de uma imagem por requisição.
- Processamento em lote de até 100 imagens, por padrão.
- Resultado individual por arquivo, mesmo quando o lote possui imagens rejeitadas.
- Validação de tipo, tamanho, resolução, integridade, domínio visual e nitidez.
- Orientação automática da imagem a partir dos dados EXIF.
- Retorno do Top 3, das probabilidades por classe, da confiança e da margem entre as duas classes mais prováveis.
- Consolidação do lote com condição predominante, consenso e identificação de heterogeneidade.
- Documentação interativa pelo Swagger UI.
- Configuração por variáveis de ambiente e imagem Docker pronta para execução.

## Classes reconhecidas

O modelo distribuído neste repositório trabalha com as seguintes classes:

| Rótulo retornado pela API | Interpretação |
|---|---|
| `Ataque_de_largata_Soja` | Indício visual de ataque de lagarta em soja. |
| `Cercospora` | Indício visual compatível com Cercospora. |
| `Doenca_de_Ferrugem_Soja` | Indício visual compatível com ferrugem da soja. |
| `Soja_Saudavel` | Imagem classificada como soja saudável. |

Os nomes acima são os rótulos técnicos do modelo e são retornados sem transformação para que clientes possam fazer o mapeamento de interface que preferirem.

## Como a análise funciona

Antes da inferência, cada arquivo passa por uma sequência de verificações:

1. Validação do arquivo enviado, do tipo MIME, do tamanho e da resolução.
2. Leitura da imagem e conversão para RGB; imagens JPG, PNG, WEBP, BMP e TIFF são aceitas.
3. Análise visual básica para verificar presença de vegetação, coerência da área verde e nitidez.
4. Redimensionamento para `300 × 300` pixels e pré-processamento compatível com EfficientNet.
5. Inferência do modelo e avaliação de confiança e margem entre as classes.

Uma imagem que não pareça conter vegetação/lavoura suficiente é marcada como `fora_do_dominio`. Imagens desfocadas são marcadas como `baixa_qualidade`. Quando a imagem é válida, mas a previsão não atinge os critérios mínimos, a API retorna `Inconclusivo` em vez de forçar uma classificação.

Na análise em lote, todas as imagens que passaram pelos filtros participam da média de probabilidades. A API também calcula a concordância entre os resultados individuais e sinaliza `Lote_heterogeneo` quando mais de uma condição confiável tem presença relevante.

## Pré-requisitos

Para executar diretamente na máquina:

- Python 3.10;
- `pip` atualizado;
- o diretório `models/` presente, incluindo o SavedModel e `classes.json`.

> O projeto utiliza TensorFlow 2.15. Para evitar incompatibilidades, use Python 3.10 conforme definido no repositório.

## Execução local

No PowerShell, a partir da raiz do projeto:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 7860
```

Com o servidor em execução, acesse:

- Swagger UI: <http://127.0.0.1:7860/docs>
- OpenAPI JSON: <http://127.0.0.1:7860/openapi.json>
- Verificação de saúde: <http://127.0.0.1:7860/health>

Em Linux ou macOS, a ativação do ambiente virtual é feita com:

```bash
source .venv/bin/activate
```

## Teste online opcional

Se quiser apenas testar a API, sem instalar dependências ou escrever código, acesse a documentação interativa do FastAPI:

- [Abrir documentação interativa (Swagger UI)](https://tccamsamericana-api-doencas-soja.hf.space/docs)

Nela, é possível enviar imagens diretamente para os endpoints `POST /predict` e `POST /predict/batch`, além de visualizar as respostas em tempo real.

Os links públicos da instância hospedada são:

| Recurso | Link |
|---|---|
| Status da API | [https://tccamsamericana-api-doencas-soja.hf.space/](https://tccamsamericana-api-doencas-soja.hf.space/) |
| Documentação interativa | [https://tccamsamericana-api-doencas-soja.hf.space/docs](https://tccamsamericana-api-doencas-soja.hf.space/docs) |
| Análise de uma imagem | [https://tccamsamericana-api-doencas-soja.hf.space/predict](https://tccamsamericana-api-doencas-soja.hf.space/predict) |
| Análise em lote | [https://tccamsamericana-api-doencas-soja.hf.space/predict/batch](https://tccamsamericana-api-doencas-soja.hf.space/predict/batch) |

> Os endpoints de análise aceitam somente requisições `POST`. Para testar pelo navegador, use a documentação interativa; os links de `/predict` e `/predict/batch` são úteis para integrações e clientes HTTP.

## Execução com Docker

Crie a imagem:

```bash
docker build -t api-doencas-soja .
```

Inicie o contêiner expondo a porta da aplicação:

```bash
docker run --rm -p 7860:7860 --name api-doencas-soja api-doencas-soja
```

Depois, a documentação estará disponível em <http://localhost:7860/docs>.

Para alterar parâmetros sem reconstruir a imagem, passe variáveis de ambiente ao iniciar o contêiner:

```bash
docker run --rm -p 7860:7860 \
  -e CONFIDENCE_THRESHOLD=0.70 \
  -e CORS_ORIGINS=http://localhost:3000 \
  --name api-doencas-soja \
  api-doencas-soja
```

## Rotas da API

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Apresenta status, versão, classes e rotas principais. |
| `GET` | `/health` | Confirma que a aplicação e o modelo foram carregados. |
| `GET` | `/model-info` | Informa arquitetura, dimensões de entrada, classes e limites configurados. |
| `POST` | `/predict` | Analisa uma imagem enviada no campo multipart `file`. |
| `POST` | `/predict/batch` | Analisa várias imagens no campo multipart repetido `files`. |
| `POST` | `/predict-batch` | Alias compatível de `/predict/batch`. |
| `GET` | `/docs` | Interface Swagger UI. |

### `GET /health`

Resposta esperada:

```json
{
  "status": "ok",
  "modelo_carregado": true,
  "quantidade_classes": 4
}
```

Use esta rota em verificações de disponibilidade, balanceadores de carga ou monitoramento de contêineres.

### `GET /model-info`

Retorna os parâmetros relevantes para o cliente, como dimensões de entrada, limites por requisição e limiares de decisão ativos. Os valores refletem a configuração atual da instância, inclusive quando variáveis de ambiente foram definidas.

### `POST /predict`

Recebe uma única imagem pelo campo obrigatório `file`, com `multipart/form-data`.

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---:|---|
| `file` | arquivo | Sim | Imagem JPG, PNG, WEBP, BMP ou TIFF. |

### `POST /predict/batch`

Recebe uma ou mais imagens pelo campo obrigatório e repetido `files`, com `multipart/form-data`.

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---:|---|
| `files` | arquivos | Sim | Uma ou mais imagens; o mesmo campo deve ser enviado para cada arquivo. |

O endpoint preserva a ordem dos arquivos no array `resultados`. Arquivos que não podem ser processados aparecem com seu respectivo status, enquanto os demais continuam sendo analisados.

## Exemplos de uso

### Uma imagem com cURL

No Windows:

```powershell
curl.exe -X POST "http://127.0.0.1:7860/predict" `
  -F "file=@C:\fotos\soja_001.jpg"
```

Em Linux ou macOS:

```bash
curl -X POST "http://127.0.0.1:7860/predict" \
  -F "file=@/caminho/para/soja_001.jpg"
```

Exemplo de resposta bem-sucedida:

```json
{
  "arquivo": "soja_001.jpg",
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

Os valores numéricos do exemplo são ilustrativos. `confianca`, `margem` e as probabilidades são retornadas em percentual. O formato de `top3` é uma lista de pares `[classe, percentual]`.

### Várias imagens com cURL

```powershell
curl.exe -X POST "http://127.0.0.1:7860/predict/batch" `
  -F "files=@C:\fotos\soja_001.jpg" `
  -F "files=@C:\fotos\soja_002.jpg" `
  -F "files=@C:\fotos\soja_003.jpg"
```

O retorno contém um identificador temporário do lote, o resumo agregado e o diagnóstico por arquivo:

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

No exemplo, `resultados` foi reduzido para manter a leitura objetiva. Em uma resposta real, ele contém o diagnóstico completo de cada imagem.

### Integração com JavaScript

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
    const erro = await resposta.json().catch(() => null);
    throw new Error(erro?.detail || "Não foi possível analisar as imagens.");
  }

  return resposta.json();
}
```

Não defina manualmente o cabeçalho `Content-Type` ao usar `FormData`: o navegador inclui automaticamente o *boundary* necessário ao envio multipart.

## Respostas e status

### Status por imagem

| Status | Resultado | Significado |
|---|---|---|
| `ok` | Nome da classe | A classificação atingiu os limiares de confiança e margem. |
| `baixa_confianca` | `Inconclusivo` | A maior probabilidade não atingiu a confiança mínima. |
| `classes_proximas` | `Inconclusivo` | As duas classes mais prováveis ficaram próximas demais. |
| `fora_do_dominio` | `Nao_e_soja` | Não houve evidência visual suficiente de vegetação/lavoura compatível. |
| `baixa_qualidade` | `Inconclusivo` | A imagem não atingiu o requisito mínimo de nitidez. |
| `arquivo_vazio` | `Inconclusivo` | O arquivo enviado está vazio. |
| `arquivo_muito_grande` | `Inconclusivo` | O arquivo ultrapassou o limite individual configurado. |
| `tipo_invalido` | `Inconclusivo` | O tipo MIME enviado não está entre os formatos aceitos. |
| `imagem_invalida` | `Inconclusivo` | O conteúdo não pôde ser lido como imagem válida. |

No endpoint unitário, erros de arquivo retornam HTTP `400`, exceto `arquivo_muito_grande`, que retorna HTTP `413`. No processamento em lote, esses casos são registrados no item correspondente; a requisição só falha quando o lote inteiro viola uma regra, por exemplo, ausência de arquivos, quantidade acima do limite ou tamanho total excedido.

### Status do lote

| Status | Resultado | Significado |
|---|---|---|
| `ok` | Nome da classe | Há confiança, margem e consenso suficientes para uma condição predominante. |
| `heterogeneo` | `Lote_heterogeneo` | Mais de uma condição confiável apresentou participação relevante. |
| `consenso_insuficiente` | `Inconclusivo` | O conjunto não apresentou confiança, margem ou consenso adequados. |
| `sem_imagens_analisaveis` | `Inconclusivo` | Nenhuma imagem passou pelos controles iniciais. |

`consenso` é a porcentagem de imagens analisadas pelo modelo cujo resultado Top-1 coincide com a condição predominante. Já `ocorrencias_confiaveis` considera somente imagens individuais com status `ok`.

## Configuração

As variáveis abaixo permitem adequar a API ao ambiente sem editar o código.

| Variável | Padrão | Descrição |
|---|---:|---|
| `MODEL_PATH` | `models/modelo_ml_savedmodel` | Caminho do diretório do SavedModel. |
| `CLASSES_PATH` | `models/classes.json` | Caminho do arquivo JSON com as classes do modelo. |
| `IMG_SIZE` | `300` | Largura e altura da entrada do modelo, em pixels. |
| `CONFIDENCE_THRESHOLD` | `0.65` | Confiança mínima para aceitar uma previsão individual ou agregada. |
| `MARGIN_THRESHOLD` | `0.18` | Diferença mínima entre as duas classes mais prováveis. |
| `BATCH_SUPPORT_THRESHOLD` | `0.50` | Consenso mínimo necessário para aceitar o resultado predominante do lote. |
| `MAX_IMAGE_SIZE_MB` | `20` | Tamanho máximo permitido por imagem. |
| `MAX_BATCH_SIZE_MB` | `500` | Tamanho máximo somado dos arquivos de um lote. |
| `MAX_BATCH_FILES` | `100` | Quantidade máxima de imagens por lote. |
| `MODEL_BATCH_SIZE` | `16` | Número de imagens enviadas ao modelo em cada bloco de inferência. |
| `MAX_IMAGE_PIXELS` | `50000000` | Quantidade máxima de pixels de uma imagem decodificada. |
| `QUALITY_MAX_EDGE` | `1600` | Maior dimensão usada nas verificações de domínio e qualidade. |
| `CORS_ORIGINS` | `*` | Origens permitidas, separadas por vírgula. |

Exemplo de configuração local no PowerShell:

```powershell
$env:CONFIDENCE_THRESHOLD = "0.70"
$env:CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:5173"
uvicorn main:app --host 0.0.0.0 --port 7860
```

> Em produção, defina `CORS_ORIGINS` com os domínios exatos do front-end. Evite manter `*` quando a API estiver exposta publicamente.

## Boas práticas para captura

A qualidade da foto influencia diretamente a utilidade da classificação. Para obter resultados mais consistentes:

- mantenha a planta ou o dossel em foco e com boa iluminação;
- evite imagens excessivamente distantes, desfocadas, contra a luz ou com forte compressão;
- faça registros em diferentes pontos do talhão, em vez de basear a decisão em uma única foto;
- em levantamentos com drone, utilize altura e resolução que preservem detalhes visuais relevantes;
- confira os resultados individuais quando o lote for heterogêneo ou inconclusivo;
- valide no campo qualquer suspeita de praga ou doença antes de definir manejo.

## Estrutura do projeto

```text
.
├── main.py                       # Aplicação FastAPI, inferência e rotas
├── domain_gate.py                # Critérios de domínio visual e qualidade
├── requirements.txt              # Dependências Python
├── Dockerfile                    # Imagem de execução da aplicação
└── models/
    ├── classes.json              # Rótulos associados à saída do modelo
    └── modelo_ml_savedmodel/     # Modelo TensorFlow SavedModel
```

## Limitações e uso responsável

- Esta API realiza classificação de imagens, não diagnóstico agronômico definitivo.
- O desempenho depende da representatividade dos dados utilizados no treinamento e das condições de captura. O repositório não inclui métricas de validação do modelo; portanto, não é apropriado inferir acurácia ou desempenho em condições não avaliadas.
- A checagem de domínio é um filtro visual simples para reduzir entradas claramente inadequadas. Ela não substitui uma validação agronômica nem garante que toda imagem aprovada seja de soja.
- Fotos aéreas muito altas podem perder detalhes de lesões durante o redimensionamento para a entrada do modelo.
- Resultados `Inconclusivo` e `Lote_heterogeneo` devem ser tratados como sinal para revisar as imagens e realizar inspeção em campo.

Para trabalhos acadêmicos ou uso operacional, registre o protocolo de captura, equipamento, altura de voo quando aplicável, área amostrada e a validação realizada por especialista. Esses dados são essenciais para interpretar os resultados de forma responsável.
