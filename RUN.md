# RUN

## Dev quickstart
```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

## TLS (dev)
```bash
hypercorn -c ../hypercorn.toml app.main:app
```

## Mongo (Docker)
```bash
docker run -d --name mongo -p 27017:27017 mongo:6
```
