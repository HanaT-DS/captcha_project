"""
scraper.py – Pipeline de scraping : detection de CAPTCHAs sur une liste d'URLs

Ouvre un navigateur, visite chaque URL, lance detect_captcha(),
sauvegarde les screenshots/HTML et genere un results.json
avec le statut de chaque page (ok, detected, error).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from playwright.sync_api import TimeoutError as PWTimeout, Page

from webscraping.browser import BrowserConfig, create_browser_and_context, close_browser
from webscraping.detect import detect_captcha


def _now_run_id() -> str:
    """Génère un identifiant de run basé sur la date/heure."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_page_title(page: Page) -> str:
    """Récupère le titre de la page sans faire planter le script."""
    try:
        return page.title() or ""
    except Exception:
        return ""


def goto_with_fallback(page: Page, url: str, timeout_ms: int = 60000) -> None:
    """
    Navigation robuste :
    - On tente d'abord domcontentloaded (souvent suffisant)
    - Si timeout, on tente commit (moins strict)
    - Si ça échoue encore, on tente load
    """
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        return
    except PWTimeout:
        pass

    try:
        page.goto(url, wait_until="commit", timeout=timeout_ms)
        return
    except PWTimeout:
        pass

    page.goto(url, wait_until="load", timeout=timeout_ms)


def save_artifacts(
    page: Page,
    out_dir: Path,
    prefix: str,
    save_html: bool = True,
    save_screenshot: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Sauvegarde les "preuves" de debug :
    - screenshot complet
    - HTML de la page
    Retourne les chemins écrits (ou None si échec).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = None
    html_path = None

    if save_screenshot:
        try:
            screenshot_path = str(out_dir / f"{prefix}_screenshot.png")
            page.screenshot(path=screenshot_path, full_page=True)
        except Exception:
            screenshot_path = None

    if save_html:
        try:
            html_path = str(out_dir / f"{prefix}_page.html")
            (out_dir / f"{prefix}_page.html").write_text(page.content(), encoding="utf-8")
        except Exception:
            html_path = None

    return {"screenshot": screenshot_path, "html": html_path}


def run_scraper(
    urls: List[str],
    headless: bool = False,
    out_base_dir: str = "runs",
    wait_after_load_ms: int = 4000,
    timeout_ms: int = 60000,
) -> Path:
    """
    Lance un run de scraping "détection uniquement".

    - Ouvre chaque URL
    - Détecte CAPTCHA / challenge
    - Sauvegarde logs JSON
    - Sauvegarde screenshot + HTML en cas de détection ou d'erreur

    Retourne le dossier du run (ex: runs/20260118_162233).
    """
    run_id = _now_run_id()
    out_dir = Path(out_base_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = BrowserConfig(headless=headless)
    pw, browser, context = create_browser_and_context(cfg)

    results: List[Dict[str, Any]] = []

    try:
        for i, url in enumerate(urls, start=1):
            prefix = f"{i:03d}"

            print("\n" + "=" * 90)
            print(f"[{i}] URL: {url}")

            page = context.new_page()

            record: Dict[str, Any] = {
                "index": i,
                "input_url": url,
                "final_url": None,
                "title": None,
                "status": None,  # ok | detected | error
                "detection": None,
                "artifacts": {"screenshot": None, "html": None},
                "error": None,
            }

            try:
                goto_with_fallback(page, url, timeout_ms=timeout_ms)
                page.wait_for_timeout(wait_after_load_ms)

                record["final_url"] = page.url
                record["title"] = _safe_page_title(page)

                # --- Détection ---
                det = detect_captcha(page)
                record["detection"] = det.to_dict()

                if det.detected:
                    record["status"] = "detected"
                    # On sauvegarde les preuves quand on détecte quelque chose
                    record["artifacts"] = save_artifacts(page, out_dir, prefix, save_html=True, save_screenshot=True)
                    print("✅ Détection:", det.category, "| provider:", det.provider)
                else:
                    record["status"] = "ok"
                    print("❌ Aucun CAPTCHA détecté")

            except Exception as e:
                record["status"] = "error"
                record["error"] = repr(e)

                # En cas d'erreur, on sauvegarde aussi les preuves si possible
                record["artifacts"] = save_artifacts(page, out_dir, prefix, save_html=True, save_screenshot=True)
                print("⚠️ Erreur:", repr(e))

            finally:
                results.append(record)
                page.close()

        # --- Sauvegarde du résumé JSON ---
        results_path = out_dir / "results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n✅ Run terminé.")
        print(f"📁 Dossier: {out_dir}")
        print(f"📄 Résultats: {results_path}")

        return out_dir

    finally:
        close_browser(pw, browser, context)
