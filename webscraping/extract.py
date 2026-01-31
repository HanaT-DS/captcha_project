"""
extract.py – Extraction de l'image CAPTCHA depuis la page

A partir du resultat de detect_captcha(), extrait l'image du CAPTCHA
via 3 methodes (base64, telechargement URL, screenshot Playwright)
et localise le champ de saisie associe (CSS selector + XPath).
Retourne un ExtractResult.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

from playwright.sync_api import Page

from webscraping.detect import CaptchaDetectionResult


# ---------------------------------------------------------------------
# 1) Résultat d'extraction
# ---------------------------------------------------------------------

@dataclass
class ExtractResult:
    ok: bool
    method: Optional[str] = None        # base64 | download | screenshot
    image_path: Optional[str] = None
    input_selector: Optional[str] = None
    input_xpath: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------
# 2) Helpers image
# ---------------------------------------------------------------------

def _save_base64_image(data_url: str, out_path: str) -> None:
    """
    Sauvegarde une image encodée en base64 (data:image/png;base64,...)
    """
    _, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    with open(out_path, "wb") as f:
        f.write(binary)


def _download_image(page: Page, url: str, out_path: str) -> None:
    """
    Télécharge une image via Playwright (session du navigateur).
    """
    response = page.request.get(url, timeout=30000)
    if not response.ok:
        raise RuntimeError(f"HTTP error while downloading image: {response.status}")
    with open(out_path, "wb") as f:
        f.write(response.body())


# ---------------------------------------------------------------------
# 3) Extraction principale
# ---------------------------------------------------------------------

def extract_captcha_image(
    page: Page,
    detection: CaptchaDetectionResult,
    out_path: str,
) -> ExtractResult:
    """
    Extrait l'image CAPTCHA + localise le champ de saisie associé.

    Méthodes image (ordre de priorité) :
    1) base64 (data:image/...)
    2) téléchargement via URL (absolue OU relative)
    3) screenshot Playwright (fallback)

    Retourne aussi :
    - selector + xpath du champ input CAPTCHA (si trouvé)
    """

    # --------------------------------------------------
    # 1) Vérifications
    # --------------------------------------------------
    img_loc = detection.captcha_image_location
    if not img_loc:
        return ExtractResult(
            ok=False,
            error="Aucune image CAPTCHA localisée",
        )

    locator = page.locator(img_loc.css_selector).first

    # --------------------------------------------------
    # 2) Extraction image
    # --------------------------------------------------
    try:
        src = locator.get_attribute("src") or ""

        # a) base64
        if src.startswith("data:image"):
            _save_base64_image(src, out_path)
            method = "base64"

        # b) téléchargement (URL absolue OU relative)
        elif src and not src.startswith("data:"):
            # Si l'URL est relative (/captcha.php...), on la rend absolue
            absolute_url = urljoin(page.url, src)

            try:
                _download_image(page, absolute_url, out_path)
                method = "download"
            except Exception:
                # Si le download échoue (403, etc.), fallback screenshot
                locator.screenshot(path=out_path)
                method = "screenshot"

        # c) fallback screenshot (src vide / cas étrange)
        else:
            locator.screenshot(path=out_path)
            method = "screenshot"

    except Exception as e:
        return ExtractResult(
            ok=False,
            error=f"Erreur extraction image: {e}",
        )

    # --------------------------------------------------
    # 3) Localisation du champ de saisie CAPTCHA
    # --------------------------------------------------
    input_selector = None
    input_xpath = None

    if detection.captcha_input_location:
        input_selector = detection.captcha_input_location.css_selector
        input_xpath = detection.captcha_input_location.xpath

    # --------------------------------------------------
    # 4) Résultat final
    # --------------------------------------------------
    return ExtractResult(
        ok=True,
        method=method,
        image_path=out_path,
        input_selector=input_selector,
        input_xpath=input_xpath,
        error=None,
    )
