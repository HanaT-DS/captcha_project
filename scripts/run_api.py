"""
run_api.py – Démarre le serveur FastAPI (mode développement)

Lance l'API de résolution de CAPTCHAs avec rechargement automatique.
L'API expose les endpoints /solve (YOLO), /solve-llm (Gemini)
et /solve-auto (cascade YOLO → Gemini).

Usage:
    uv run python -m scripts.run_api
"""

import uvicorn

if __name__ == "__main__":
    # Tu peux override via variables d'environnement si besoin
    # os.environ["YOLO_MODEL_PATH"] = "models/weights/best.pt"
    # os.environ["YOLO_DATA_YAML"] = "datasets/Yolo-Captcha-Test-7/data.yaml"

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # recharge auto quand tu modifies les fichiers (dev)
    )
