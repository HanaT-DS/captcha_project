"""
test_fill.py – Test end-to-end : detect → extract → solve (cascade) → fill & submit

Pipeline complet :
1. Ouvre le navigateur sur une URL avec CAPTCHA texte
2. Détecte le CAPTCHA
3. Extrait l'image du CAPTCHA
4. Résout le CAPTCHA en cascade : YOLO finetuned d'abord → Gemini fallback si vide ou confiance basse
5. Tape le texte dans le champ input comme un humain
6. Soumet le formulaire

Usage:
    uv run python -m scripts.test_fill
    uv run python -m scripts.test_fill --solver cascade        (défaut)
    uv run python -m scripts.test_fill --solver gemini          (force Gemini uniquement)
    uv run python -m scripts.test_fill --solver yolo            (force YOLO uniquement)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from playwright.sync_api import TimeoutError as PWTimeout

from webscraping.browser import BrowserConfig, create_browser_and_context, close_browser
from webscraping.detect import detect_captcha
from webscraping.extract import extract_captcha_image
from webscraping.fill import fill_and_submit


URL = "https://www.metropolegrandparis.fr/fr/formulaire-de-contact"


# ----------------------------
# Pretty printers
# ----------------------------

def pretty_print_detection(result):
    print("\n📌 RÉSULTAT DE DÉTECTION")
    print("-" * 60)

    if not result.detected:
        print("❌ Aucun CAPTCHA détecté")
        return

    print("✅ CAPTCHA détecté")
    print(f"Type        : {result.category}")
    print(f"Provider    : {result.provider}")
    print(f"Score       : {result.score}")

    if result.captcha_image_location:
        loc = result.captcha_image_location
        print(f"\n🖼️ Image CAPTCHA :")
        print(f"  CSS   : {loc.css_selector}")
        print(f"  XPath : {loc.xpath}")

    if result.captcha_input_location:
        loc = result.captcha_input_location
        print(f"\n⌨️ Champ de saisie :")
        print(f"  CSS   : {loc.css_selector}")
        print(f"  XPath : {loc.xpath}")


def pretty_print_extract(result):
    print("\n🧪 RÉSULTAT D'EXTRACTION")
    print("-" * 60)

    if not result.ok:
        print("❌ Extraction échouée")
        if result.error:
            print(f"Erreur : {result.error}")
        return

    print("✅ Extraction réussie")
    print(f"Méthode      : {result.method}")
    print(f"Image        : {result.image_path}")

    if result.input_selector or result.input_xpath:
        print(f"\n⌨️ Champ de saisie :")
        if result.input_selector:
            print(f"  CSS   : {result.input_selector}")
        if result.input_xpath:
            print(f"  XPath : {result.input_xpath}")


def pretty_print_solve(text: str, meta: dict):
    print("\n🤖 RÉSULTAT DU SOLVE")
    print("-" * 60)

    if not text:
        print("❌ Aucun texte résolu")
        return

    print(f"✅ Texte résolu : {text}")
    print(f"Méthode         : {meta.get('method', '?')}")
    if meta.get("yolo_text") is not None:
        print(f"YOLO texte      : {meta['yolo_text']}")
    if meta.get("yolo_avg_conf") is not None:
        print(f"YOLO conf moy.  : {meta['yolo_avg_conf']}")
    if meta.get("fallback"):
        print(f"Fallback        : oui ({meta.get('fallback_reason', '?')})")
    else:
        print(f"Fallback        : non")
    if meta.get("warning"):
        print(f"⚠️ Warning      : {meta['warning']}")


def pretty_print_fill(result):
    print("\n✍️ RÉSULTAT DU FILL & SUBMIT")
    print("-" * 60)

    if not result.ok:
        print("❌ Saisie échouée")
        if result.error:
            print(f"Erreur : {result.error}")
        return

    print("✅ Saisie réussie")
    print(f"Texte tapé      : {result.text_typed}")
    print(f"Submit cliqué   : {result.submit_clicked}")


# ----------------------------
# Solvers
# ----------------------------

CASCADE_THRESHOLD = 0.70  # seuil confiance moyenne YOLO → fallback Gemini


def _yolo_avg_conf(detections: list) -> float:
    if not detections:
        return 0.0
    return sum(d["conf"] for d in detections) / len(detections)


def solve_cascade(image_path: str) -> tuple[str, dict]:
    """
    Stratégie cascade : YOLO finetuned d'abord → Gemini fallback
    si texte vide ou confiance moyenne < seuil.
    """
    from models.yolo_ocr import YoloOcrEngine, get_model_config
    from models.llm_ocr import LlmOcrEngine, DEFAULT_MODEL

    meta = {"method": "cascade"}

    # --- Phase 1 : YOLO finetuned ---
    print("  ↳ Phase 1 : YOLO finetuned_v1...")
    try:
        cfg = get_model_config("finetuned_v1")
        yolo = YoloOcrEngine(
            model_path=cfg["weights"],
            data_yaml_path=cfg["data_yaml"],
            conf=cfg.get("conf", 0.25),
            imgsz=cfg.get("imgsz", 640),
            stretch_to=cfg.get("stretch_to"),
        )
        yolo_out = yolo.solve(image_path=image_path, verbose=False)
        yolo_text = yolo_out["text"]
        avg_conf = _yolo_avg_conf(yolo_out["detections"])
        meta["yolo_text"] = yolo_text
        meta["yolo_avg_conf"] = round(avg_conf, 4)
        print(f"    YOLO → '{yolo_text}' (conf moy: {avg_conf:.2f})")
    except Exception as exc:
        yolo_text = ""
        avg_conf = 0.0
        meta["yolo_text"] = ""
        meta["yolo_avg_conf"] = 0.0
        meta["yolo_error"] = str(exc)
        print(f"    YOLO → erreur: {exc}")

    # --- Décision fallback ---
    need_fallback = False
    if not yolo_text:
        need_fallback = True
        meta["fallback_reason"] = "empty_text"
    elif avg_conf < CASCADE_THRESHOLD:
        need_fallback = True
        meta["fallback_reason"] = "low_confidence"

    if not need_fallback:
        meta["method"] = "yolo"
        meta["fallback"] = False
        return yolo_text, meta

    # --- Phase 2 : Gemini fallback ---
    print(f"  ↳ Phase 2 : Gemini fallback ({meta.get('fallback_reason')})...")
    try:
        llm = LlmOcrEngine()
        llm_out = llm.solve(image_path=image_path)
        meta["method"] = "yolo+gemini"
        meta["fallback"] = True
        meta["gemini_model"] = DEFAULT_MODEL
        print(f"    Gemini → '{llm_out['text']}'")
        return llm_out["text"], meta
    except Exception as exc:
        meta["gemini_error"] = str(exc)
        print(f"    Gemini → erreur: {exc}")
        # Si YOLO avait quand même un texte, on le retourne
        if yolo_text:
            meta["method"] = "yolo"
            meta["fallback"] = True
            meta["warning"] = "Gemini fallback échoué, résultat YOLO retourné"
            return yolo_text, meta
        meta["method"] = "failed"
        meta["fallback"] = True
        return "", meta


def solve_gemini_only(image_path: str) -> tuple[str, dict]:
    from models.llm_ocr import LlmOcrEngine, DEFAULT_MODEL
    engine = LlmOcrEngine()
    result = engine.solve(image_path=image_path)
    return result["text"], {"method": "gemini", "gemini_model": DEFAULT_MODEL, "fallback": False}


def solve_yolo_only(image_path: str) -> tuple[str, dict]:
    from models.yolo_ocr import YoloOcrEngine, get_model_config
    cfg = get_model_config("finetuned_v1")
    engine = YoloOcrEngine(
        model_path=cfg["weights"],
        data_yaml_path=cfg["data_yaml"],
        conf=cfg.get("conf", 0.25),
        imgsz=cfg.get("imgsz", 640),
        stretch_to=cfg.get("stretch_to"),
    )
    result = engine.solve(image_path=image_path, verbose=False)
    text = result["text"]
    avg_conf = _yolo_avg_conf(result["detections"])
    return text, {
        "method": "yolo",
        "yolo_text": text,
        "yolo_avg_conf": round(avg_conf, 4),
        "fallback": False,
    }


# ----------------------------
# Helpers
# ----------------------------

def goto_with_fallback(page, url: str) -> None:
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

    page.goto(url, wait_until="load", timeout=60000)


def save_debug_artifacts(page, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        page.screenshot(path=str(out_dir / "page_screenshot.png"), full_page=True)
    except Exception:
        pass

    try:
        (out_dir / "page.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Test end-to-end : detect → extract → solve → fill")
    parser.add_argument(
        "--solver",
        choices=["cascade", "gemini", "yolo"],
        default="cascade",
        help="Stratégie de résolution (défaut: cascade = YOLO → Gemini fallback)",
    )
    parser.add_argument("--url", default=URL, help=f"URL à tester (défaut: {URL})")
    parser.add_argument("--no-submit", action="store_true", help="Ne pas soumettre le formulaire")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"fill_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = BrowserConfig(headless=False)
    pw, browser, context = create_browser_and_context(cfg)

    page = context.new_page()
    try:
        # --- Navigation ---
        print("\n🌐 PAGE")
        print("-" * 60)
        print(f"URL : {args.url}")

        goto_with_fallback(page, args.url)
        page.wait_for_timeout(4000)

        print(f"URL finale : {page.url}")
        try:
            print(f"Titre      : {page.title()}")
        except Exception:
            pass

        # --- 1) Détection ---
        det = detect_captcha(page)
        pretty_print_detection(det)

        if not det.detected:
            print("\n⚠️ Pas de CAPTCHA détecté, arrêt du test.")
            return

        # On accepte tout CAPTCHA qui a une image + un champ input (text, captcha_widget, etc.)
        if not det.captcha_image_location:
            print(f"\n⚠️ CAPTCHA de type '{det.category}' détecté mais pas d'image CAPTCHA localisée — impossible de solve.")
            return

        if not det.captcha_input_location:
            print(f"\n⚠️ CAPTCHA détecté mais pas de champ input localisé — impossible de fill.")
            return

        # --- 2) Extraction ---
        captcha_path = str(out_dir / "captcha.png")
        ext = extract_captcha_image(page, det, captcha_path)
        pretty_print_extract(ext)

        if not ext.ok:
            print("\n⚠️ Extraction échouée, arrêt du test.")
            return

        # --- 3) Solve (cascade par défaut) ---
        print(f"\n⏳ Résolution avec stratégie '{args.solver}'...")
        solver_fn = {
            "cascade": solve_cascade,
            "gemini": solve_gemini_only,
            "yolo": solve_yolo_only,
        }[args.solver]
        solved_text, solve_meta = solver_fn(ext.image_path)
        pretty_print_solve(solved_text, solve_meta)

        if not solved_text:
            print("\n⚠️ Texte vide après résolution, arrêt du test.")
            return

        # --- 4) Fill & Submit ---
        print(f"\n⏳ Saisie du texte '{solved_text}' dans le champ...")

        # Screenshot avant saisie
        try:
            page.screenshot(path=str(out_dir / "before_fill.png"), full_page=True)
        except Exception:
            pass

        fill_result = fill_and_submit(
            page=page,
            input_selector=ext.input_selector or "",
            text=solved_text,
            input_xpath=ext.input_xpath,
            submit=not args.no_submit,
        )
        pretty_print_fill(fill_result)

        # Screenshot après saisie/soumission
        page.wait_for_timeout(2000)
        try:
            page.screenshot(path=str(out_dir / "after_fill.png"), full_page=True)
        except Exception:
            pass

        # --- Sauvegarde des résultats ---
        save_debug_artifacts(page, out_dir)

        results = {
            "url": args.url,
            "strategy": args.solver,
            "detection": det.to_dict(),
            "extraction": ext.__dict__,
            "solve": {"text": solved_text, **solve_meta},
            "fill": {
                "ok": fill_result.ok,
                "text_typed": fill_result.text_typed,
                "submit_clicked": fill_result.submit_clicked,
                "error": fill_result.error,
            },
        }
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\n📁 Fichiers sauvegardés dans : {out_dir}")

    except Exception as e:
        print(f"\n💥 Erreur inattendue : {e}")
        save_debug_artifacts(page, out_dir)

    finally:
        input("\nEntrée pour fermer le navigateur...")
        close_browser(pw, browser, context)


if __name__ == "__main__":
    main()
