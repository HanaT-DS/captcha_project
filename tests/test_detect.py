"""
test_detect.py – Test d'integration : detection de CAPTCHAs sur des URLs reelles

Ouvre un navigateur Playwright sur une liste d'URLs de test,
lance detect_captcha() sur chacune, affiche les resultats
(type, provider, score, preuves) et sauvegarde les screenshots/HTML.

Usage:
    uv run python -m tests.test_detect
"""

from pathlib import Path
from datetime import datetime

from playwright.sync_api import TimeoutError as PWTimeout

from webscraping.browser import (
    BrowserConfig,
    create_browser_and_context,
    close_browser,
)
from webscraping.detect import detect_captcha


TEST_URLS = [
    # "https://2captcha.com/fr/demo/normal", # CAPTCHA text
     "https://www.metropolegrandparis.fr/fr/formulaire-de-contact", # CAPTCHA text
    # "https://mon.aphp.fr/sso/app/account", # ca marche pas 
    # "https://auth.service-public.gouv.fr/realms/service-public/protocol/openid-connect/auth?response_type=code&client_id=spclient&scope=address%20phone%20openid%20profile%20email&state=HnhIBPkkcHP9ziujfuW5246imfXV1GXn3Y9BkPX_kMg%3D&redirect_uri=https://www.service-public.gouv.fr/openid_connect_login&nonce=7cDp9OgCi7MAxvLQcJUstub2IEXvxwCzdh7btnRpVuo", # CAPTCHA text
    # "https://connect.france-visas.gouv.fr/realms/usager/login-actions/registration?client_id=fv-fo-keycloak-web&tab_id=XdV6Tt5F8-w&client_data=eyJydSI6Imh0dHBzOi8vYXBwbGljYXRpb24tZm9ybS5mcmFuY2UtdmlzYXMuZ291di5mci9mdi1mby1kZGUvbG9naW4vb2F1dGgyL2NvZGUva2V5Y2xvYWsiLCJydCI6ImNvZGUiLCJzdCI6Iml4ajk0aTltSno0cXpMS0gtYnpNazRjZHYxNkdNR2tmZ1FPeUhNVEdybkU9In0", # CAPTCHA text *
     # veut pas marcher car pas de cookies
    # "https://www.w3schools.in/demo/phptextcaptcha/demo.php",
    # "https://captcha.com/demos/features/captcha-demo.aspx",
    # "https://metropole.toulouse.fr/nous-contacter", # reCAPTCHA
    # "https://www.google.com/recaptcha/api2/demo", # reCAPTCHA
    # "https://demo.turnstile.workers.dev/", # turnstile
    # "https://2captcha.com/demo/cloudflare-turnstile", # turnstil
    # "https://www.reddit.com",



    # "https://accounts.hcaptcha.com/demo",
    # "https://friendlycaptcha.com/demo",
    # "https://example.com",
]


def save_debug_artifacts(page, out_dir: Path, prefix: str) -> None:
    """
    Sauvegarde des preuves utiles pour debug/rapport :
    - screenshot complet
    - HTML de la page (si possible)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Screenshot
    try:
        page.screenshot(path=str(out_dir / f"{prefix}_screenshot.png"), full_page=True)
    except Exception:
        pass

    # HTML
    try:
        html = page.content()
        (out_dir / f"{prefix}_page.html").write_text(html, encoding="utf-8")
    except Exception:
        pass


def goto_with_fallback(page, url: str) -> None:
    """
    Navigation robuste :
    1) on tente domcontentloaded
    2) si timeout, on tente commit (moins strict)
    3) si encore timeout, on tente load
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        return
    except PWTimeout:
        pass

    try:
        page.goto(url, wait_until="commit", timeout=60000)
        return
    except PWTimeout:
        pass

    # Dernier essai
    page.goto(url, wait_until="load", timeout=60000)


def pretty_print_result(result):
    """
    Affichage lisible et pédagogique du résultat de detect_captcha.
    """
    print("\n📌 RÉSULTAT DE DÉTECTION")
    print("-" * 60)

    if not result.detected:
        print("❌ Aucun CAPTCHA détecté")
        return

    print(f"✅ CAPTCHA détecté")
    print(f"Type        : {result.category}")
    print(f"Provider    : {result.provider}")
    if getattr(result, "visibility", None):
        print(f"Visibilité  : {result.visibility}")
    print(f"Score       : {result.score}")

    # ---------------------------
    # Widget provider
    # ---------------------------
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

    # ---------------------------
    # CAPTCHA textuel (OCR)
    # ---------------------------
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

    # ---------------------------
    # Page de challenge
    # ---------------------------
    if result.challenge:
        ch = result.challenge
        print("\n🚧 Page de CHALLENGE :")
        print(f"  URL finale : {ch.final_url}")
        print(f"  Title     : {ch.title}")
        if ch.matched_keywords:
            print(f"  Mots-clés : {', '.join(ch.matched_keywords)}")
        if ch.matched_url_hints:
            print(f"  URL hints : {', '.join(ch.matched_url_hints)}")

    # ---------------------------
    # Preuves
    # ---------------------------
    print("\n🔎 Preuves détectées :")
    for e in result.evidences:
        provider = f" ({e.provider_guess})" if e.provider_guess else ""
        print(f"  - [{e.kind}] {e.value}{provider}")


def main():
    # Dossier de sortie daté (pratique pour le rapport)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / run_id

    cfg = BrowserConfig(headless=False)
    pw, browser, context = create_browser_and_context(cfg)

    for i, url in enumerate(TEST_URLS, start=1):
        print("\n" + "=" * 90)
        print(f"[{i}] Testing URL: {url}")

        page = context.new_page()
        prefix = f"{i:03d}"

        try:
            # Navigation robuste
            goto_with_fallback(page, url)

            # Petite pause pour laisser le DOM se stabiliser (widgets/captcha)
            page.wait_for_timeout(5000)

            # Détection
            result = detect_captcha(page)
            print("Final URL:", page.url)
            #print(result.to_dict())
            pretty_print_result(result)


            # Si captcha/challenge détecté => on garde les preuves
            if result.detected:
                save_debug_artifacts(page, out_dir, prefix)

        except Exception as e:
            print("Error while loading/detecting:", repr(e))

            # On essaye quand même de sauver ce qu’on peut
            save_debug_artifacts(page, out_dir, prefix)

        finally:
            page.close()

    input("\nAppuie sur Entrée pour fermer le navigateur...")
    close_browser(pw, browser, context)


if __name__ == "__main__":
    main()
