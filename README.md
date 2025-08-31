# Narval Cart — Structured API (Mongo + CV)

Routes (as requested):
- **POST /relateds** — create product relations
- **GET /relateds** — list relations
- **DELETE /relateds/{id}** — delete a relation
- **POST /items** — create cart item
- **GET /items** — list cart items
- **DELETE /items/{id}** — delete cart item
- **POST /frames** — send a frame; returns detected products in ROI and related suggestions for products in cart

MongoDB collections:
- `relateds`: `{ product: string, related: string, score: number }`
- `products` (support): `{ type: string, name: string, path: string }`

## Run
```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# .env (optional)
cat > .env << 'EOF'
API_PREFIX=/v1
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=narvalcart
ROI=100,100,400,400
MATCH_THRESHOLD=0.78
TEMPLATES_DIR=app/assets/templates
USE_ONNX=false
ONNX_MODEL_PATH=app/assets/models/detection.onnx
ONNX_MODEL_DATA=app/assets/models/model.data
EOF

uvicorn app.main:app --reload --port 8080
# http://localhost:8080/v1/openapi.json
```

## Examples (cURL)

### Create relation (relateds)
```bash
curl -s -X POST 'http://localhost:8080/v1/relateds'  -H 'Content-Type: application/json'  -d '[{"product":"macarrão","related":"molho de tomate","score":0.9}]' | jq .
```

### List relations by product
```bash
curl -s 'http://localhost:8080/v1/relateds?page=1&limit=50&product=macarrão' | jq .
```

### Create cart item
```bash
curl -s -X POST 'http://localhost:8080/v1/items'  -H 'Content-Type: application/json'  -d '{"name":"macarrão","quantity":2,"note":"grano duro"}' | jq .
```

### Send frame (ROI detection + suggestions)
```bash
# create a base64 of your frame.png
b64=$(base64 -w 0 frame.png)  # macOS: base64 frame.png | tr -d '\n'

curl -s -X POST 'http://localhost:8080/v1/frames'  -H 'Content-Type: application/json'  -d '{
  "frame": { "encoding":"base64", "data":"'"$b64"'" },
  "cartProductIds": ["macarrão"]
}' | jq .
```
