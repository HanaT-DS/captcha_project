# Captcha Project

Webscraping robuste aux CAPTCHAs : detection multi-types, extraction et resolution automatique des CAPTCHAs texte avec saisie dans le formulaire.

Le module de detection identifie plusieurs types de CAPTCHAs (reCAPTCHA, hCaptcha, Turnstile, CAPTCHAs texte, pages de challenge), mais la resolution automatique cible les **CAPTCHAs texte** (image + champ de saisie). Elle combine un modele YOLO (detection de caracteres) entraine sur des CAPTCHAs synthetiques puis fine-tune sur des CAPTCHAs reels, avec un fallback Gemini (vision LLM), le tout expose via une API FastAPI et orchestre par Playwright.

## Architecture

```
                    +-------------------+
                    |   run_captcha     |
                    |   _solver.py      |   Script de production
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |   webscraping/  |          |    API FastAPI   |
     |                 |          |    (api/)        |
     | browser.py      |  HTTP   |                  |
     | detect.py       +-------->+ /solve      YOLO |
     | extract.py      |         | /solve-llm  LLM  |
     | fill.py         |         | /solve-auto      |
     +-----------------+         +--------+---------+
                                          |
                                 +--------v--------+
                                 |    models/       |
                                 |                  |
                                 | yolo_ocr.py      |
                                 | llm_ocr.py       |
                                 | weights/         |
                                 +-----------------+
```

**L'API sert de pont** entre le webscraping et les modeles : le script envoie l'image extraite a l'API via HTTP, qui gere l'inference et retourne le texte resolu.

## Types de CAPTCHAs detectes

Le module `detect.py` identifie automatiquement :

| Type | Detection | Resolution |
|------|-----------|------------|
| **CAPTCHA texte** (image + input) | Oui | **Oui** (YOLO + Gemini) |
| reCAPTCHA (v2, v3) | Oui | Non |
| hCaptcha | Oui | Non |
| Cloudflare Turnstile | Oui | Non |
| Pages de challenge | Oui | Non |

## Pipeline de resolution (CAPTCHAs texte)

```
1. Naviguer vers l'URL cible (Playwright)
2. Fermer les bandeaux de cookies (tarteaucitron, Ezoic, generiques)
3. Detecter le CAPTCHA (analyse DOM : iframes, selecteurs, mots-cles, URLs)
4. Extraire l'image (base64 / download / screenshot)
5. Resoudre via l'API :
   - Cascade (defaut) : YOLO finetuned -> Gemini fallback
   - Fallback si : texte vide, < 4 caracteres, ou confiance < 0.70
6. Taper le texte caractere par caractere (simulation humaine)
7. Soumettre le formulaire
8. Verifier le resultat (message de succes / redirection / re-detection)
9. Si echec -> retry (jusqu'a max-retries)
```

## Entrainement du modele

### Phase 1 : Baseline (CAPTCHAs synthetiques)

- **Dataset** : [Yolo-Captcha-Test-7](https://universe.roboflow.com/) sur Roboflow
  - 2 415 images train / 674 valid / 340 test
  - 62 classes (0-9, a-z, A-Z)
- **Modele** : YOLOv8s (small) entraine 120 epochs sur Google Colab (NVIDIA L4)
- **Augmentations** : rotation 15deg, scale 0.4, shear 5, perspective — mosaic/mixup desactives (melangent les caracteres)
- **Grid search** sur le validation set : meilleure config `imgsz=640, conf=0.15, iou=0.6`
- **Notebook** : `notebooks/yolo_baseline.ipynb`

| Metrique | Sans normalisation | Avec normalisation I/l, O/0 |
|----------|-------------------|----------------------------|
| Exact Match (test) | 86.5% (294/340) | 91.2% (310/340) |
| CER moyen | 2.7% | 1.7% |

> Les principales confusions sont I/l (19 cas) — la normalisation les corrige automatiquement.

### Phase 2 : Finetuning (CAPTCHAs reels)

- **Site cible** : metropolegrandparis.fr (formulaire de contact)
- **Collecte** : 305 CAPTCHAs scrapes automatiquement (`scripts/data_prep/01_scrape_captchas.py`)
- **Annotation** : 150 images annotees manuellement sur Roboflow, puis un modele YOLO intermediaire entraine sur ces 150 a aide a pre-annoter le reste
- **Dataset final** : exporte via l'API Roboflow — 230 train / 50 valid / 25 test
- **Transfer learning** : a partir du modele baseline, lr0=0.001 (10x plus petit), freeze=10 couches, 100 epochs, patience=15
- **Notebook** : `notebooks/yolo_finetuning.ipynb`

| Metrique | Baseline (Roboflow) | Finetuned v1 (reel) |
|----------|--------------------|--------------------|
| Exact Match (test) | 86.5% | 84.0% (21/25) |
| CER moyen | 2.7% | 3.2% |

> Les datasets sont differents (synthetique vs reel), la comparaison est indicative. Le finetuned performe bien sur les CAPTCHAs du site cible malgre un test set reduit (25 images).

### Phase 3 : Fallback Gemini (LLM vision)

Quand YOLO n'est pas assez confiant, l'API bascule automatiquement sur Gemini :
- Texte YOLO vide
- Moins de 4 caracteres detectes
- Confiance moyenne < 0.70

## Structure du projet

```
captcha-project/
|-- api/                              # API FastAPI (pont modeles <-> webscraping)
|   |-- main.py                       # Endpoints /solve, /solve-llm, /solve-auto
|   +-- schemas.py                    # Modeles Pydantic
|
|-- models/                           # Moteurs d'inference
|   |-- yolo_ocr.py                   # YoloOcrEngine + MODEL_REGISTRY
|   |-- llm_ocr.py                    # LlmOcrEngine (Gemini vision)
|   +-- weights/                      # Poids des modeles (.pt)
|
|-- webscraping/                      # Automatisation navigateur (Playwright)
|   |-- browser.py                    # BrowserConfig, create/close browser
|   |-- detect.py                     # detect_captcha() — analyse DOM
|   |-- extract.py                    # extract_captcha_image() — base64/download/screenshot
|   +-- fill.py                       # fill_and_submit() — saisie humaine + submit
|
|-- scripts/                          # Scripts de production
|   |-- data_prep/                    # Preparation des donnees YOLO
|   |   |-- 01_scrape_captchas.py     # Collecte automatique de CAPTCHAs
|   |   +-- 02_auto_annotate.py       # Pre-annotation avec le modele baseline
|   |-- run_api.py                    # Demarrer l'API FastAPI
|   +-- run_captcha_solver.py         # Pipeline complet end-to-end
|
|-- tests/                            # Scripts de test
|   |-- test_detect.py                # Test detection sur URLs reelles
|   |-- test_extract.py               # Test extraction d'image
|   |-- test_fill.py                  # Test pipeline detect -> extract -> solve -> fill
|   |-- test_run_ocr.py               # Test YOLO sur une image
|   +-- test_run_llm.py               # Test Gemini sur une image
|
+-- notebooks/                        # Notebooks d'entrainement (Google Colab)
    |-- yolo_baseline.ipynb           # Entrainement baseline sur donnees Roboflow
    +-- yolo_finetuning.ipynb         # Finetuning sur CAPTCHAs reels
```

## Installation

```bash
# Cloner le projet
git clone <repo-url>
cd captcha-project

# Installer les dependances (Python >= 3.12)
uv sync

# Installer Playwright
uv run playwright install chromium
```

## Configuration

Creer un fichier `.env` a la racine :

```env
GOOGLE_API_KEY=...          # Cle API Google (pour Gemini vision)
ROBOFLOW_API_KEY=...        # Cle API Roboflow (pour export dataset)
```

## Utilisation

### 1. Demarrer l'API

```bash
uv run python -m scripts.run_api
```

L'API est accessible sur `http://localhost:8000` (docs Swagger sur `/docs`).

### 2. Lancer le solver

```bash
# Mode cascade (defaut) — YOLO finetuned + Gemini fallback
uv run python -m scripts.run_captcha_solver \
    --url "https://www.metropolegrandparis.fr/fr/formulaire-de-contact"

# Mode visible (debug)
uv run python -m scripts.run_captcha_solver \
    --url "https://www.metropolegrandparis.fr/fr/formulaire-de-contact" \
    --no-headless

# Sans soumission (tape le texte + screenshot, sans submit)
uv run python -m scripts.run_captcha_solver \
    --url "..." --no-headless --no-submit

# Forcer un solver specifique
uv run python -m scripts.run_captcha_solver --url "..." --solver yolo
uv run python -m scripts.run_captcha_solver --url "..." --solver gemini
```

### 3. Endpoints API

| Methode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/solve` | Resolution YOLO (modele specifique) |
| POST | `/solve-llm` | Resolution Gemini (vision) |
| POST | `/solve-auto` | Cascade YOLO -> Gemini (recommande) |
| GET | `/models` | Liste des modeles disponibles |
| GET | `/health` | Healthcheck |

### 4. Preparer des donnees

```bash
# Scraper 300 CAPTCHAs depuis un site
uv run python -m scripts.data_prep.01_scrape_captchas \
    --url "https://2captcha.com/fr/demo/normal" --count 300

# Pre-annoter avec le modele baseline
uv run python -m scripts.data_prep.02_auto_annotate
```

## Technologies

- **Python 3.12**
- **YOLOv8** (Ultralytics) — detection de caracteres
- **Google Gemini** (vision) — fallback LLM
- **FastAPI** + Uvicorn — API REST
- **Playwright** — automatisation navigateur
- **Roboflow** — annotation et gestion de datasets
- **Google Colab** (NVIDIA L4) — entrainement GPU
