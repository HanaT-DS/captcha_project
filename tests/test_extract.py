"""
test_extract.py – Test d'integration : detection + extraction d'image CAPTCHA

Ouvre un navigateur sur une URL de test, detecte le CAPTCHA,
puis extrait l'image (base64, download ou screenshot) et localise
le champ de saisie associe. Sauvegarde les resultats en JSON.

Usage:
    uv run python -m tests.test_extract
"""

from pathlib import Path
from datetime import datetime
import json

from webscraping.browser import BrowserConfig, create_browser_and_context, close_browser
from webscraping.detect import detect_captcha
from webscraping.extract import extract_captcha_image

URL = "https://www.metropolegrandparis.fr/fr/formulaire-de-contact"

def pretty_print_detection(result):
    """
    Affichage lisible et pédagogique du résultat de detect_captcha.
    """
    print("\n📌 RÉSULTAT DE DÉTECTION")
    print("-" * 60)

    if not result.detected:
        print("❌ Aucun CAPTCHA détecté")
        return

    print("✅ CAPTCHA détecté")
    print(f"Type        : {result.category}")
    print(f"Provider    : {result.provider}")

    # Si tu as ajouté visibility dans detect.py, ça s'affichera ici
    if getattr(result, "visibility", None):
        print(f"Visibilité  : {result.visibility}")

    print(f"Score       : {result.score}")

    # Widget provider (reCAPTCHA / Turnstile / hCaptcha...)
    if result.widget_location:
        loc = result.widget_location
        print("\n🧩 Widget CAPTCHA :")
        print(f"  CSS   : {loc.css_selector}")
        print(f"  XPath : {loc.xpath}")
        if loc.bbox:
            print(f"  BBox  : {loc.bbox}")

    if result.iframe_location:
        loc = result.iframe_location
        print("\n🖼️ Iframe CAPTCHA :")
        print(f"  CSS   : {loc.css_selector}")
        print(f"  XPath : {loc.xpath}")
        if loc.bbox:
            print(f"  BBox  : {loc.bbox}")

    # CAPTCHA textuel (OCR)
    if result.captcha_image_location:
        loc = result.captcha_image_location
        print("\n🖼️ Image CAPTCHA (OCR) :")
        print(f"  CSS   : {loc.css_selector}")
        print(f"  XPath : {loc.xpath}")
        if loc.bbox:
            print(f"  BBox  : {loc.bbox}")

    if result.captcha_input_location:
        loc = result.captcha_input_location
        print("\n⌨️ Champ de saisie CAPTCHA :")
        print(f"  CSS   : {loc.css_selector}")
        print(f"  XPath : {loc.xpath}")
        if loc.bbox:
            print(f"  BBox  : {loc.bbox}")

    # Challenge page (si jamais)
    if result.challenge:
        ch = result.challenge
        print("\n🚧 Page de CHALLENGE :")
        print(f"  URL finale : {ch.final_url}")
        print(f"  Title     : {ch.title}")
        if ch.matched_keywords:
            print(f"  Mots-clés : {', '.join(ch.matched_keywords)}")
        if ch.matched_url_hints:
            print(f"  URL hints : {', '.join(ch.matched_url_hints)}")

    # Preuves
    print("\n🔎 Preuves détectées :")
    for e in result.evidences:
        prov = f" ({e.provider_guess})" if e.provider_guess else ""
        print(f"  - [{e.kind}] {e.value}{prov}")


def pretty_print_extract(result):
    """
    Affichage lisible et pédagogique du résultat d'extraction.
    """
    print("\n🧪 RÉSULTAT D'EXTRACTION")
    print("-" * 60)

    if not result.ok:
        print("❌ Extraction échouée")
        if result.error:
            print(f"Erreur : {result.error}")
        return

    print("✅ Extraction réussie")
    print(f"Méthode utilisée : {result.method}")
    print(f"Fichier image    : {result.image_path}")

    if result.method == "base64":
        print("Détails          : image encodée en base64 dans le HTML (qualité parfaite)")
    elif result.method == "download":
        print("Détails          : image téléchargée via l'URL du src")
    elif result.method == "screenshot":
        print("Détails          : capture de l'élément <img> (fallback)")

    # ✅ NOUVEAU : afficher le champ input associé (si disponible)
    if getattr(result, "input_selector", None) or getattr(result, "input_xpath", None):
        print("\n⌨️ Champ de saisie (retourné par extract.py) :")
        if result.input_selector:
            print(f"  CSS   : {result.input_selector}")
        if result.input_xpath:
            print(f"  XPath : {result.input_xpath}")


def save_debug_artifacts(page, out_dir: Path) -> None:
    """
    Sauvegarde des preuves utiles pour ton rapport :
    - screenshot complet
    - HTML de la page
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        page.screenshot(path=str(out_dir / "page_screenshot.png"), full_page=True)
    except Exception:
        pass

    try:
        (out_dir / "page.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = BrowserConfig(headless=False)
    pw, browser, context = create_browser_and_context(cfg)

    page = context.new_page()
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)

        # ✅ NOUVEAU : afficher URL finale + titre (utile si redirections)
        try:
            title = page.title()
        except Exception:
            title = ""
        print("\n🌐 PAGE")
        print("-" * 60)
        print("URL finale :", page.url)
        print("Titre     :", title)

        # 1) Détection
        det = detect_captcha(page)
        pretty_print_detection(det)

        # Sauvegarde des preuves de page (utile pour rapport)
        save_debug_artifacts(page, out_dir)

        # ✅ NOUVEAU : sauvegarder le résultat de détection en JSON
        (out_dir / "detection_result.json").write_text(
            json.dumps(det.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 2) Extraction si image CAPTCHA localisée
        out_path = str(out_dir / "captcha.png")
        res = extract_captcha_image(page, det, out_path)
        pretty_print_extract(res)

        # ✅ NOUVEAU : sauvegarder le résultat d'extraction en JSON
        (out_dir / "extract_result.json").write_text(
            json.dumps(res.__dict__, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n📁 Fichiers sauvegardés dans :", out_dir)

    finally:
        input("\nEntrée pour fermer...")
        close_browser(pw, browser, context)


if __name__ == "__main__":
    main()
