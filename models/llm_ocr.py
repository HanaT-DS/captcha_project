"""
llm_ocr.py – Moteur OCR basé sur Gemini (vision LLM)

Utilise la capacité vision de Gemini (google-genai) pour lire
le texte directement depuis l'image du CAPTCHA.
Sert de fallback quand YOLO n'est pas assez confiant.

Fournit :
- LlmOcrEngine : classe d'inférence (solve)
- DEFAULT_MODEL : modèle Gemini par défaut
- Nécessite GOOGLE_API_KEY dans l'environnement.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


# ----------------------------
# Configuration Gemini
# ----------------------------

DEFAULT_MODEL = "gemini-2.5-pro"
ENV_KEY = "GOOGLE_API_KEY"

# Prompt envoyé au LLM pour lire le texte de l'image
CAPTCHA_TEXT_PROMPT = (
    "Act as a blind person assistant. "
    "Read the text from the image and give me only the text answer. "
    "No quotes, no explanation, no punctuation, just the raw characters."
)


# ----------------------------
# Utils
# ----------------------------

def _guess_mime_type(image_path: str | Path) -> str:
    """Devine le type MIME à partir de l'extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def _clean_llm_response(raw: str) -> str:
    """
    Nettoie la réponse brute du LLM.
    Supprime les espaces, guillemets, backticks, retours à la ligne, etc.
    """
    text = raw.strip()
    # Supprimer les backticks markdown (```text ... ```)
    text = re.sub(r"^```\w*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    # Supprimer guillemets englobants
    text = text.strip("\"'` \n\r\t")
    return text


# ----------------------------
# Solver Gemini
# ----------------------------

def solve_with_gemini(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    prompt: str | None = None,
) -> str:
    """
    Envoie l'image du CAPTCHA à Google Gemini (vision) et retourne le texte lu.

    Le fonctionnement :
    1. Lit les bytes bruts de l'image
    2. Envoie au LLM sous forme de Part binaire (pas besoin de base64)
    3. Le LLM analyse l'image et retourne le texte
    4. Retourne le texte brut nettoyé
    """
    from google import genai
    from google.genai import types

    key = api_key or os.getenv(ENV_KEY)
    if not key:
        raise ValueError(f"{ENV_KEY} non définie (env ou paramètre)")

    client = genai.Client(api_key=key)
    mime = _guess_mime_type(image_path)
    user_prompt = prompt or CAPTCHA_TEXT_PROMPT

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            user_prompt,
        ],
    )

    raw = response.text or ""
    return _clean_llm_response(raw)


# ----------------------------
# Point d'entrée fonctionnel
# ----------------------------

def solve_captcha_text(
    image_path: str | Path,
    model: str | None = None,
    api_key: str | None = None,
    prompt: str | None = None,
) -> str:
    """
    Résout un CAPTCHA texte via Gemini.

    Args:
        image_path: Chemin vers l'image du CAPTCHA
        model: Modèle à utiliser (défaut: gemini-2.5-pro)
        api_key: Clé API (optionnel, sinon prise depuis .env)
        prompt: Prompt custom (optionnel)

    Returns:
        Le texte lu dans le CAPTCHA
    """
    return solve_with_gemini(
        image_path=image_path,
        model=model or DEFAULT_MODEL,
        api_key=api_key,
        prompt=prompt,
    )


# ============================================================
# Classe LlmOcrEngine (même interface que YoloOcrEngine)
# ============================================================

class LlmOcrEngine:
    """
    Moteur OCR basé sur Gemini (vision).
    Même interface solve() que YoloOcrEngine pour être interchangeable.

    Note: les LLMs ne retournent pas de bounding boxes, donc
    detections sera toujours une liste vide.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
    ):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key
        self.prompt = prompt

        # Vérifier que la clé API est disponible
        effective_key = api_key or os.getenv(ENV_KEY)
        if not effective_key:
            raise ValueError(
                f"Clé API manquante. "
                f"Définis {ENV_KEY} dans .env ou passe api_key="
            )

    def solve(
        self,
        image_path: str | Path,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Résout un CAPTCHA texte via Gemini.
        Retourne le même format que YoloOcrEngine.solve() :
        - text: le texte lu
        - detections: [] (pas de boxes avec les LLMs)
        """
        text = solve_captcha_text(
            image_path=image_path,
            model=self.model,
            api_key=self.api_key,
            prompt=self.prompt,
        )

        return {
            "text": text,
            "detections": [],
            "image_path": str(image_path),
        }
