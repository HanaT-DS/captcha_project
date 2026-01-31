"""
test_run_llm.py – Test unitaire Gemini : résout un CAPTCHA via LlmOcrEngine

Charge le moteur Gemini (vision) et prédit le texte d'une image
CAPTCHA donnée. Nécessite GOOGLE_API_KEY dans l'environnement.

Usage:
    uv run python -m tests.test_run_llm --image "path/to/captcha.png"
    uv run python -m tests.test_run_llm --image "path/to/captcha.png" --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models.llm_ocr import LlmOcrEngine, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Résout un CAPTCHA texte via Gemini (vision)"
    )
    parser.add_argument("--image", required=True, help="Chemin vers l'image du CAPTCHA")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Modèle Gemini (défaut: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image introuvable: {args.image}")

    model_name = args.model or DEFAULT_MODEL
    print(f"Provider : gemini ({model_name})")

    engine = LlmOcrEngine(model=args.model)
    result = engine.solve(image_path=args.image)

    print(f"Texte    : {result['text']}")


if __name__ == "__main__":
    main()

# Exemples:
# uv run python -m tests.test_run_llm --image "datasets/captcha_target_v1/raw/captcha_0005.png"
# uv run python -m tests.test_run_llm --image "datasets/captcha_target_v1/raw/captcha_0005.png" --model gemini-2.5-flash
