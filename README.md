# Narval Cart API

API para detecção e recomendação de produtos em carrinhos de compra. Criando relação entre esses produtos.
para gerar um fluxo de compra mais eficiente.

Componentes principais:

- **Items**: cadastro de itens (nome + imagem) com *slug* normalizado.
- **Relateds**: relações “produto → relacionado” com *score* (ex.: macarrão → molho de tomate).
- **Frames**: ingestão de quadros (arquivo) para detectar produtos em **área de interesse** usando um modelo de visão computacional.
- **HTTP/2**: suportamos abrir conexão via HTTP/2 (recomendado para **melhor streaming de frames**).

> Na raiz do projeto há uma coleção do Postman para facilitar os testes: **`doc-postman-requests.json`**.

---

## Sumário

- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Configuração (.env)](#configuração-env)
- [Execução — Local (venv)](#execução--local-venv)
- [Execução — Docker Compose](#execução--docker-compose)
- [HTTP/2 (Hypercorn)](#http2-hypercorn)
- [Coleção Postman](#coleção-postman)
- [Endpoints](#endpoints)
- [Exemplos de uso (curl)](#exemplos-de-uso-curl)
- [Equipe](#equipe)
- [Licença](#licença)

---

## Arquitetura

- **FastAPI** (ASGI)  
- **MongoDB** (collections: `items`, `relateds`, `products` de suporte) via **Motor** (async)  
- **Jinja2** para views HTML de sugestão  
- **OpenCV** (headless) opcional para decodificar frames e rodar o modelo de detecção
- **Hypercorn** (HTTP/2)
- **uvicorn** (HTTP/1.1) 
- **uvloop** (loop de eventos)
- **uvicorn-reload** (reload)
- **YOLO** para detecção de ROI
- **Clip** para definição melhor de tipos de objetos
- **Arquivos estáticos** das imagens servidos em `/static` (diretório configurável)

---

## Pré-requisitos

- **Python 3.12+**
- **MongoDB 6/7/8+** (local ou via container)
- **lib de sistema**: `build-essential` pode ser necessário para algumas wheels

---

## Configuração (.env)

Crie um arquivo **`.env`** na raiz ou em `app/.env` com:

```env
API_PREFIX=/v1
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=narvalcart
UPLOAD_DIR=app/assets/uploads
PUBLIC_BASE_URL=http://localhost:8080
