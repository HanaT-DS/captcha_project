"""
run_captcha_solver.py – Pipeline complet de résolution de CAPTCHA

Automatise : naviguer → détecter → extraire → résoudre (via API) → remplir → soumettre
avec retry automatique si le CAPTCHA est mal résolu.

L'API sert de pont entre le webscraping et les modèles de résolution.
Le script envoie l'image extraite à l'API, qui gère l'inférence
(YOLO, Gemini, cascade).

Prérequis:
    L'API doit être démarrée : uv run uvicorn api.main:app --port 8000

Usage:
    uv run python -m scripts.run_captcha_solver --url "https://example.com/contact"
    uv run python -m scripts.run_captcha_solver --url "..." --no-headless --no-submit
    uv run python -m scripts.run_captcha_solver --url "..." --solver yolo --max-retries 3
    uv run python -m scripts.run_captcha_solver --url "..." --record  # enregistre un GIF du solve
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import httpx

from playwright.sync_api import TimeoutError as PWTimeout

from webscraping.browser import BrowserConfig, create_browser_and_context, close_browser
from webscraping.detect import detect_captcha
from webscraping.extract import extract_captcha_image
from webscraping.fill import fill_and_submit
from webscraping.gif_recorder import GifRecorder


# ============================================================
# Solvers via API
# ============================================================

def _solve_via_api(
    image_path: str,
    api_url: str,
    endpoint: str,
    params: dict | None = None,
) -> tuple[str, dict]:
    """
    Envoie l'image à l'API et retourne (text, meta).
    """
    url = f"{api_url.rstrip('/')}{endpoint}"

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/png")}
        response = httpx.post(url, files=files, params=params or {}, timeout=60.0)

    if response.status_code != 200:
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text
        raise RuntimeError(f"API error {response.status_code}: {error_detail}")

    data = response.json()
    text = data.get("text", "")
    meta = data.get("meta", {})
    return text, meta


def make_solver(solver_name: str, api_url: str) -> Callable:
    """
    Retourne une fonction solver qui appelle l'endpoint API approprié.

    - cascade → POST /solve-auto  (YOLO finetuned → Gemini fallback)
    - yolo    → POST /solve        (YOLO finetuned uniquement)
    - gemini  → POST /solve-llm   (Gemini uniquement)
    """
    if solver_name == "cascade":
        def solve(image_path: str) -> tuple[str, dict]:
            # /solve-auto retourne déjà "method" dans meta (yolo | yolo+gemini)
            return _solve_via_api(image_path, api_url, "/solve-auto")
        return solve

    elif solver_name == "yolo":
        def solve(image_path: str) -> tuple[str, dict]:
            text, meta = _solve_via_api(
                image_path, api_url, "/solve",
                params={"model": "finetuned_v1"},
            )
            meta.setdefault("method", "yolo")
            return text, meta
        return solve

    elif solver_name == "gemini":
        def solve(image_path: str) -> tuple[str, dict]:
            text, meta = _solve_via_api(image_path, api_url, "/solve-llm")
            meta.setdefault("method", "gemini")
            return text, meta
        return solve

    else:
        raise ValueError(f"Solver inconnu: {solver_name}")


def check_api_health(api_url: str) -> bool:
    """Vérifie que l'API est accessible."""
    try:
        resp = httpx.get(f"{api_url.rstrip('/')}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# Navigation robuste
# ============================================================

def goto_with_fallback(page, url: str) -> None:
    for wait_until in ["domcontentloaded", "commit", "load"]:
        try:
            page.goto(url, wait_until=wait_until, timeout=60000)
            return
        except PWTimeout:
            if wait_until == "load":
                raise


def dismiss_cookie_banners(page) -> None:
    """
    Ferme les bandeaux de cookies courants pour éviter qu'ils
    bloquent les clics sur le formulaire.
    Supporte : tarteaucitron, Ezoic, CMP génériques.
    """
    # 1) Essayer de cliquer un bouton connu (refuser/accepter)
    selectors = [
        # tarteaucitron
        "#tarteaucitronAllDenied2",
        "#tarteaucitronAllAllowed2",
        "button.tarteaucitronDeny",
        # Ezoic
        "#ez-accept-all",
        "#ez-cookie-dialog-wrapper button[class*='accept']",
        # Génériques
        "[id*='cookie'] button[class*='deny']",
        "[id*='cookie'] button[class*='reject']",
        "[id*='cookie'] button[class*='accept']",
        "button[id*='cookie-accept']",
        "[class*='cookie-banner'] button",
        "[class*='consent'] button[class*='accept']",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if btn.count() and btn.is_visible():
                btn.click(timeout=3000)
                page.wait_for_timeout(500)
                return
        except Exception:
            continue

    # 2) Fallback JS : supprimer les overlays connus qui bloquent les clics
    try:
        page.evaluate("""(() => {
            const ids = [
                '#tarteaucitronRoot',
                '#ez-cmpv2-container',
                '#ez-cookie-dialog-wrapper',
                '[class*="cookie-banner"]',
                '[class*="cookie-consent"]',
            ];
            ids.forEach(sel => document.querySelector(sel)?.remove());
        })()""")
    except Exception:
        pass


# ============================================================
# Vérification post-submit
# ============================================================

_SUCCESS_SELECTORS = [
    "[class*='success']",
    "[class*='Success']",
    "[class*='alert-success']",
    "[class*='message-ok']",
    "[class*='confirmation']",
    ".flash-success",
    "[role='alert']",
]

_SUCCESS_WORDS = [
    "success", "succès", "réussi", "passed", "correct",
    "envoyé", "sent", "thank", "merci",
]


def _detect_success_message(page) -> bool:
    """
    Cherche un message de succès après soumission.
    1) Cherche des éléments HTML typiques de messages de succès (class=success, alert, etc.)
       et vérifie qu'ils contiennent un mot-clé de succès.
    2) Évite de chercher dans tout le body (qui peut contenir "error" dans du code/doc).
    """
    for sel in _SUCCESS_SELECTORS:
        try:
            loc = page.locator(sel)
            for i in range(loc.count()):
                el = loc.nth(i)
                if not el.is_visible():
                    continue
                text = el.inner_text().lower()
                if any(w in text for w in _SUCCESS_WORDS):
                    return True
        except Exception:
            continue
    return False


# ============================================================
# Pipeline (une tentative)
# ============================================================

def run_pipeline(
    page,
    url: str,
    solver_fn: Callable,
    out_dir: Path,
    attempt: int,
    wait_ms: int,
    submit: bool = True,
    recorder: Optional[GifRecorder] = None,
) -> dict:
    """
    Exécute une tentative complète : detect → extract → solve (via API) → fill.
    Si submit=True  : soumet le formulaire puis vérifie si le CAPTCHA est résolu.
    Si submit=False : tape le texte sans soumettre, prend un screenshot et s'arrête.
    Retourne un dict {success, text, method, error}.
    """
    prefix = f"attempt_{attempt}"

    # 1) Navigation
    goto_with_fallback(page, url)
    page.wait_for_timeout(wait_ms)

    # 1b) Fermer les bandeaux de cookies (tarteaucitron, etc.)
    dismiss_cookie_banners(page)

    # Frame : page chargee (CAPTCHA visible)
    if recorder:
        recorder.capture(page, duration_ms=1000)

    # 2) Détection
    det = detect_captcha(page)
    if not det.detected:
        print("    Pas de CAPTCHA détecté — page accessible directement.")
        return {"success": True, "text": None, "method": "no_captcha", "error": None}

    if not det.captcha_image_location:
        return {"success": False, "text": None, "method": None,
                "error": f"CAPTCHA '{det.category}' détecté mais pas d'image localisée"}

    if not det.captcha_input_location:
        return {"success": False, "text": None, "method": None,
                "error": "CAPTCHA détecté mais pas de champ input localisé"}

    print(f"    CAPTCHA détecté (type: {det.category}, score: {det.score})")

    # 3) Extraction
    captcha_path = str(out_dir / f"{prefix}_captcha.png")
    ext = extract_captcha_image(page, det, captcha_path)
    if not ext.ok:
        return {"success": False, "text": None, "method": None,
                "error": f"Extraction échouée: {ext.error}"}

    print(f"    Image extraite ({ext.method})")

    # Frame : CAPTCHA visible avant saisie
    if recorder:
        recorder.capture(page, duration_ms=800)

    # 4) Solve via API
    solved_text, solve_meta = solver_fn(ext.image_path)
    if not solved_text:
        return {"success": False, "text": None, "method": solve_meta.get("method"),
                "error": "Résolution échouée: texte vide"}

    print(f"    Résolu: '{solved_text}' (méthode: {solve_meta.get('method', '?')})")

    # 5) Fill (+ Submit si demandé)
    on_frame = (lambda p, d: recorder.capture(p, d)) if recorder else None
    fill_result = fill_and_submit(
        page=page,
        input_selector=ext.input_selector or "",
        text=solved_text,
        input_xpath=ext.input_xpath,
        submit=submit,
        on_frame=on_frame,
    )
    if not fill_result.ok:
        return {"success": False, "text": solved_text, "method": solve_meta.get("method"),
                "error": f"Saisie échouée: {fill_result.error}"}

    # -- Mode no-submit : on tape le texte, screenshot, et on s'arrête --
    if not submit:
        print(f"    Texte tapé (sans soumission)")
        page.wait_for_timeout(500)
        try:
            page.screenshot(path=str(out_dir / f"{prefix}_after_fill.png"), full_page=True)
        except Exception:
            pass
        return {"success": True, "text": solved_text, "method": solve_meta.get("method"),
                "error": None, **solve_meta}

    # -- Mode submit : soumettre puis vérifier --
    print(f"    Texte tapé + formulaire soumis")

    # 6) Vérifier le résultat (attendre puis analyser la page)
    page.wait_for_timeout(random.randint(2000, 3500))

    # Frame : etat final apres submit
    if recorder:
        recorder.capture(page, duration_ms=2000)

    # Screenshot post-submit
    try:
        page.screenshot(path=str(out_dir / f"{prefix}_after_submit.png"), full_page=True)
    except Exception:
        pass

    # 6a) Chercher un message de succès sur la page
    if _detect_success_message(page):
        print(f"    Message de succès détecté sur la page")
        return {"success": True, "text": solved_text, "method": solve_meta.get("method"),
                "error": None, **solve_meta}

    # 6b) Vérifier si l'URL a changé (redirection post-submit = souvent succès)
    current_url = page.url
    if current_url != url:
        print(f"    Redirection détectée → {current_url}")
        return {"success": True, "text": solved_text, "method": solve_meta.get("method"),
                "error": None, **solve_meta}

    # 6c) Re-détecter le CAPTCHA
    recheck = detect_captcha(page)
    if not recheck.detected or not recheck.captcha_image_location:
        # CAPTCHA disparu → succès
        return {"success": True, "text": solved_text, "method": solve_meta.get("method"),
                "error": None, **solve_meta}
    else:
        # CAPTCHA encore là → mauvaise réponse
        print(f"    CAPTCHA encore présent → mauvaise réponse")
        return {"success": False, "text": solved_text, "method": solve_meta.get("method"),
                "error": "CAPTCHA toujours présent après soumission (mauvaise réponse)"}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet : détection → extraction → résolution (via API) → saisie de CAPTCHA"
    )
    parser.add_argument("--url", required=True, help="URL cible avec CAPTCHA")
    parser.add_argument("--max-retries", type=int, default=5, help="Nombre max de tentatives (défaut: 5)")
    parser.add_argument(
        "--solver", choices=["cascade", "gemini", "yolo"], default="cascade",
        help="Stratégie de résolution (défaut: cascade = YOLO → Gemini fallback)",
    )
    parser.add_argument("--no-headless", action="store_true", help="Ouvrir le navigateur visible")
    parser.add_argument("--no-submit", action="store_true",
                        help="Ne pas soumettre le formulaire (tape le texte CAPTCHA + screenshot, sans retry)")
    parser.add_argument("--out-dir", default=None, help="Dossier de sortie (défaut: runs/solve_<timestamp>)")
    parser.add_argument("--wait-ms", type=int, default=4000, help="Attente après chargement en ms (défaut: 4000)")
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="URL de l'API de résolution (défaut: http://localhost:8000)",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Enregistrer un GIF du solve (sauvegardé dans successful_solves/ si succès)",
    )
    parser.add_argument(
        "--gif-scale", type=float, default=0.5,
        help="Facteur d'échelle des frames GIF (défaut: 0.5 = 640x360)",
    )
    args = parser.parse_args()

    # Dossier de sortie
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"solve_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier que l'API est accessible
    print(f"\n🔗 Vérification de l'API ({args.api_url})...")
    if not check_api_health(args.api_url):
        print(f"❌ API inaccessible à {args.api_url}")
        print(f"   Démarre l'API d'abord : uv run uvicorn api.main:app --port 8000")
        return

    print(f"✅ API accessible")

    solver_fn = make_solver(args.solver, args.api_url)

    print(f"\n🚀 CAPTCHA Solver")
    print("=" * 60)
    do_submit = not args.no_submit
    max_retries = 1 if not do_submit else args.max_retries

    print(f"URL          : {args.url}")
    print(f"Solver       : {args.solver}")
    print(f"API          : {args.api_url}")
    print(f"Submit       : {do_submit}")
    print(f"Max retries  : {max_retries}")
    print(f"Headless     : {not args.no_headless}")
    print(f"Record GIF   : {args.record}")
    print(f"Output       : {out_dir}")
    print("=" * 60)

    # Ouvrir navigateur
    cfg = BrowserConfig(headless=not args.no_headless)
    pw, browser, context = create_browser_and_context(cfg)
    page = context.new_page()

    # Recorder GIF (optionnel)
    recorder = GifRecorder(scale=args.gif_scale) if args.record else None

    results_log = []
    final_result = None

    try:
        for attempt in range(1, max_retries + 1):
            print(f"\n[Tentative {attempt}/{max_retries}]")

            # Vider les frames de la tentative précédente
            if recorder:
                recorder.discard()

            try:
                result = run_pipeline(
                    page=page,
                    url=args.url,
                    solver_fn=solver_fn,
                    out_dir=out_dir,
                    attempt=attempt,
                    wait_ms=args.wait_ms,
                    submit=do_submit,
                    recorder=recorder,
                )
                results_log.append({"attempt": attempt, **result})

                if result["success"]:
                    final_result = result

                    # Sauvegarder le GIF si succès avec texte résolu
                    if recorder and result.get("text"):
                        solves_dir = Path("successful_solves")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        gif_name = f"solve_{timestamp}_{result['text']}.gif"
                        gif_name = "".join(
                            c if c.isalnum() or c in "._-" else "_"
                            for c in gif_name
                        )
                        gif_path = recorder.save(solves_dir / gif_name)
                        if gif_path:
                            print(f"    🎬 GIF enregistré: {gif_path}")
                            result["gif_path"] = gif_path

                    break

                print(f"    ❌ {result['error']}")

            except Exception as exc:
                print(f"    💥 Erreur: {exc}")
                results_log.append({"attempt": attempt, "success": False, "error": str(exc)})

            # Pause avant retry
            if attempt < max_retries:
                delay = random.uniform(1.0, 2.5)
                page.wait_for_timeout(int(delay * 1000))

        # Résumé final
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ")
        print("=" * 60)

        if final_result and final_result["success"]:
            if final_result.get("text"):
                print(f"✅ CAPTCHA résolu avec succès")
                print(f"Texte     : {final_result['text']}")
                print(f"Méthode   : {final_result['method']}")
            else:
                print(f"✅ Pas de CAPTCHA sur cette page")
        else:
            print(f"❌ Échec après {len(results_log)} tentative(s)")
            if results_log:
                last = results_log[-1]
                print(f"Dernière erreur : {last.get('error', '?')}")

        print(f"Tentatives : {len(results_log)}")
        print(f"Artifacts  : {out_dir}")

        # Sauvegarder les résultats
        summary = {
            "url": args.url,
            "solver": args.solver,
            "api_url": args.api_url,
            "success": bool(final_result and final_result["success"]),
            "total_attempts": len(results_log),
            "attempts": results_log,
        }
        (out_dir / "results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    finally:
        page.close()
        close_browser(pw, browser, context)


if __name__ == "__main__":
    main()
