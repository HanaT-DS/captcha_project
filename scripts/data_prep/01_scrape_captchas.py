"""
01_scrape_captchas.py – Scrape N CAPTCHAs uniques depuis un site de test.

Réutilise les modules webscraping/ (browser, detect, extract, scraper).

Usage:
    uv run python -m scripts.data_prep.01_scrape_captchas \
        --url "https://2captcha.com/fr/demo/normal" \
        --count 300
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from pathlib import Path

from tqdm import tqdm

from webscraping.browser import BrowserConfig, create_browser_and_context, close_browser
from webscraping.scraper import goto_with_fallback
from webscraping.detect import detect_captcha
from webscraping.extract import extract_captcha_image


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_COUNT = 300
DEFAULT_OUT_DIR = "datasets/captcha_target_v1/raw"
DELAY_MIN = 0.5
DELAY_MAX = 1.0
WAIT_AFTER_LOAD_MS = 2000


# ── Helpers ───────────────────────────────────────────────────────────

def _md5(path: Path) -> str:
    """Hash MD5 d'un fichier."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_existing(out_dir: Path) -> tuple[int, set[str]]:
    """
    Charge les images déjà présentes (pour reprise après interruption).
    Retourne (nombre_existant, set_de_hashes_md5).
    """
    existing = sorted(out_dir.glob("captcha_*.png"))
    seen: set[str] = set()
    for f in existing:
        seen.add(_md5(f))
    return len(existing), seen


# ── Scraping principal ────────────────────────────────────────────────

def scrape_captchas(
    url: str,
    count: int = DEFAULT_COUNT,
    out_dir: str = DEFAULT_OUT_DIR,
    headless: bool = True,
    delay_min: float = DELAY_MIN,
    delay_max: float = DELAY_MAX,
) -> Path:
    """Scrape `count` CAPTCHAs uniques depuis `url`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Reprise si images déjà présentes
    saved, seen_hashes = _load_existing(out)
    if saved:
        print(f"Reprise : {saved} images existantes, {len(seen_hashes)} hashes uniques")

    duplicates = 0
    errors = 0
    max_attempts = count * 10
    attempts = 0

    cfg = BrowserConfig(headless=headless)
    pw, browser, context = create_browser_and_context(cfg)

    try:
        page = context.new_page()
        pbar = tqdm(total=count, initial=saved, desc="CAPTCHAs", unit="img")

        while saved < count and attempts < max_attempts:
            attempts += 1
            try:
                # 1) Naviguer vers la page (chaque chargement = nouveau CAPTCHA)
                goto_with_fallback(page, url)
                page.wait_for_timeout(WAIT_AFTER_LOAD_MS)

                # 2) Détecter le CAPTCHA
                det = detect_captcha(page)
                if not det.detected or not det.captcha_image_location:
                    errors += 1
                    pbar.set_postfix(dup=duplicates, err=errors)
                    time.sleep(random.uniform(delay_min, delay_max))
                    continue

                # 3) Extraire l'image (fichier temporaire)
                tmp = out / f"_tmp_{int(time.time() * 1000)}.png"
                result = extract_captcha_image(page, det, str(tmp))
                if not result.ok:
                    errors += 1
                    pbar.set_postfix(dup=duplicates, err=errors)
                    time.sleep(random.uniform(delay_min, delay_max))
                    continue

                # 4) Vérifier doublon via MD5
                h = _md5(tmp)
                if h in seen_hashes:
                    duplicates += 1
                    tmp.unlink(missing_ok=True)
                    pbar.set_postfix(dup=duplicates, err=errors)
                    time.sleep(random.uniform(delay_min, delay_max))
                    continue

                # 5) Sauvegarder avec le bon numéro séquentiel
                saved += 1
                final = out / f"captcha_{saved:04d}.png"
                tmp.rename(final)
                seen_hashes.add(h)

                pbar.update(1)
                pbar.set_postfix(dup=duplicates, err=errors)

            except Exception as e:
                errors += 1
                tqdm.write(f"Erreur: {e}")

            # 6) Délai aléatoire entre requêtes
            time.sleep(random.uniform(delay_min, delay_max))

        pbar.close()
        page.close()

        if saved < count:
            print(f"\n⚠️ Arrêt: trop d'échecs/doublons ({attempts} tentatives pour {saved} images).")

    finally:
        close_browser(pw, browser, context)

    # Résumé final
    print(f"\nImages sauvegardées : {saved}/{count}")
    print(f"Doublons ignorés    : {duplicates}")
    print(f"Erreurs             : {errors}")
    print(f"Dossier             : {out.resolve()}")

    return out


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape des CAPTCHAs depuis un site de test"
    )
    parser.add_argument(
        "--url", required=True,
        help="URL de la page avec le CAPTCHA",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Nombre d'images à collecter (défaut: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Dossier de sortie (défaut: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--no-headless", action="store_true",
        help="Afficher le navigateur (debug)",
    )
    parser.add_argument(
        "--delay-min", type=float, default=DELAY_MIN,
        help=f"Délai min entre requêtes en sec (défaut: {DELAY_MIN})",
    )
    parser.add_argument(
        "--delay-max", type=float, default=DELAY_MAX,
        help=f"Délai max entre requêtes en sec (défaut: {DELAY_MAX})",
    )
    args = parser.parse_args()

    scrape_captchas(
        url=args.url,
        count=args.count,
        out_dir=args.out_dir,
        headless=not args.no_headless,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
    )


if __name__ == "__main__":
    main()
