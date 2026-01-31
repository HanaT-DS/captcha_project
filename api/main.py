"""
main.py – API FastAPI de résolution de CAPTCHAs (multi-méthodes)

Expose les endpoints :
- POST /solve       : résolution via YOLO (détection de caractères)
- POST /solve-llm   : résolution via Gemini (vision LLM)
- POST /solve-auto  : cascade YOLO → Gemini fallback (si vide, trop peu
                       de caractères ou confiance basse)
- GET  /models      : liste des modèles YOLO disponibles
- GET  /config      : configuration courante
- GET  /health      : healthcheck

Sert de pont entre les modèles (models/) et le webscraping (scripts/).
"""

from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from api.schemas import (
    SolveResponse, ConfigResponse, ModelsResponse, ModelInfo,
)
from models.yolo_ocr import YoloOcrEngine, MODEL_REGISTRY, list_models, get_model_config
from models.llm_ocr import LlmOcrEngine, DEFAULT_MODEL as LLM_DEFAULT_MODEL, ENV_KEY as LLM_ENV_KEY


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = os.getenv("YOLO_DEFAULT_MODEL", "baseline")
DEFAULT_IOU = float(os.getenv("YOLO_IOU", "0.6"))
DEFAULT_NORMALIZE = os.getenv("YOLO_NORMALIZE", "false").lower() == "true"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(3 * 1024 * 1024)))  # 3 MB
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

# Cascade /solve-auto : seuils pour déclencher le fallback Gemini
DEFAULT_CASCADE_MODEL = os.getenv("CASCADE_YOLO_MODEL", "finetuned_v1")
DEFAULT_CASCADE_THRESHOLD = float(os.getenv("CASCADE_CONF_THRESHOLD", "0.70"))
DEFAULT_CASCADE_MIN_CHARS = int(os.getenv("CASCADE_MIN_CHARS", "4"))


app = FastAPI(
    title="Captcha Solver API",
    version="3.0.0",
    description="API OCR multi-méthodes : YOLO (détection de caractères) + LLM (Gemini vision).",
)

# Dictionnaire des engines chargés : nom -> YoloOcrEngine
engines: Dict[str, YoloOcrEngine] = {}


# ============================================================
# Startup: charger tous les modèles du registre
# ============================================================

@app.on_event("startup")
def startup_load_models():
    for name in list_models():
        cfg = get_model_config(name)
        weights = Path(cfg["weights"])
        data_yaml = Path(cfg["data_yaml"])

        if not weights.exists():
            print(f"[WARN] Modèle '{name}' ignoré: {weights} introuvable")
            continue
        if not data_yaml.exists():
            print(f"[WARN] Modèle '{name}' ignoré: {data_yaml} introuvable")
            continue

        engines[name] = YoloOcrEngine(
            model_path=weights,
            data_yaml_path=data_yaml,
            conf=cfg["conf"],
            imgsz=cfg["imgsz"],
            iou_threshold=DEFAULT_IOU,
            normalize_confusions=DEFAULT_NORMALIZE,
            stretch_to=cfg.get("stretch_to"),
            verbose=False,
        )
        print(f"[OK] Modèle '{name}' chargé ({cfg['description']})")

    if not engines:
        raise RuntimeError("Aucun modèle n'a pu être chargé")


# ============================================================
# Endpoints utilitaires
# ============================================================

@app.get("/health")
def health():
    if not engines:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ok",
        "yolo_models_loaded": list(engines.keys()),
        "gemini_available": bool(os.getenv(LLM_ENV_KEY)),
    }


@app.get("/models", response_model=ModelsResponse)
def get_models():
    """Liste les modèles disponibles dans le registre."""
    models = []
    for name in list_models():
        cfg = get_model_config(name)
        models.append(ModelInfo(
            name=name,
            description=cfg["description"],
            imgsz=cfg["imgsz"],
            conf=cfg["conf"],
            loaded=name in engines,
        ))
    return ModelsResponse(default=DEFAULT_MODEL, models=models)


@app.get("/config", response_model=ConfigResponse)
def get_config():
    """Config du modèle par défaut."""
    if DEFAULT_MODEL not in engines:
        raise HTTPException(status_code=503, detail="Default model not loaded")
    e = engines[DEFAULT_MODEL]
    return ConfigResponse(
        model_path=str(e.model_path),
        data_yaml_path=str(e.data_yaml_path),
        default_conf=e.conf,
        default_imgsz=e.imgsz,
        default_iou=e.iou_threshold,
        normalize_confusions=e.normalize_confusions,
    )


# ============================================================
# Endpoint principal: /solve
# ============================================================

@app.post("/solve", response_model=SolveResponse)
async def solve(
    file: UploadFile = File(..., description="Image (png/jpg/webp)"),
    model: Optional[str] = Query(default=None, description="Nom du modèle (voir /models)"),
    conf: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Seuil confiance"),
    imgsz: Optional[int] = Query(default=None, ge=256, le=2048, description="Taille d'entrée YOLO"),
    iou: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="IoU suppression doublons"),
    normalize: Optional[bool] = Query(default=None, description="Normaliser confusions (I/l/1, O/0)"),
):
    """
    Reçoit une image, applique YOLO OCR, retourne texte + detections.
    Paramètre `model` pour choisir le modèle (défaut: baseline).
    """
    # Résoudre le modèle
    model_name = model or DEFAULT_MODEL
    if model_name not in engines:
        loaded = list(engines.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{model_name}' non disponible. Chargés: {loaded}",
        )
    engine = engines[model_name]

    # 1) vérifier content-type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    # 2) lire le contenu + limiter taille
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max allowed: {MAX_UPLOAD_BYTES} bytes.",
        )

    # 3) écrire dans un fichier temporaire
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name


    try:
        # 4) solve
        out = engine.solve(
            image_path=tmp_path,
            conf=conf,
            imgsz=imgsz,
            iou_threshold=iou,
            normalize_confusions=normalize,
            verbose=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return SolveResponse(
            text=out["text"],
            detections=out["detections"],
            meta={
                "model": model_name,
                "filename": file.filename,
                "content_type": file.content_type,
                "elapsed_ms": round(dt_ms, 2),
                "used_conf": engine.conf if conf is None else conf,
                "used_imgsz": engine.imgsz if imgsz is None else imgsz,
                "used_iou": engine.iou_threshold if iou is None else iou,
                "used_normalize": engine.normalize_confusions if normalize is None else normalize,
            },
        )
    finally:
        # 5) nettoyage fichier temporaire
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ============================================================
# Endpoint LLM (Gemini)
# ============================================================

@app.post("/solve-llm", response_model=SolveResponse)
async def solve_llm(
    file: UploadFile = File(..., description="Image (png/jpg/webp)"),
    model: Optional[str] = Query(default=None, description="Modèle Gemini (ex: gemini-2.5-pro)"),
):
    """
    Résout un CAPTCHA texte via Gemini (vision).
    Pas besoin de conf/imgsz/iou : le LLM lit l'image directement.
    """
    # 1) Vérifier clé API
    if not os.getenv(LLM_ENV_KEY):
        raise HTTPException(
            status_code=503,
            detail=f"Clé API manquante. Définis {LLM_ENV_KEY} dans .env",
        )

    # 2) Vérifier content-type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    # 3) Lire le contenu + limiter taille
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max allowed: {MAX_UPLOAD_BYTES} bytes.",
        )

    # 4) Écrire dans un fichier temporaire
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 5) Solve via Gemini
        actual_model = model or LLM_DEFAULT_MODEL
        engine = LlmOcrEngine(model=actual_model)
        out = engine.solve(image_path=tmp_path)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return SolveResponse(
            text=out["text"],
            detections=[],
            meta={
                "method": "llm",
                "provider": "gemini",
                "model": actual_model,
                "filename": file.filename,
                "content_type": file.content_type,
                "elapsed_ms": round(dt_ms, 2),
            },
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ============================================================
# Endpoint cascade : /solve-auto (YOLO → Gemini fallback)
# ============================================================

def _yolo_avg_confidence(detections: list) -> float:
    """Confiance moyenne des détections YOLO, ou 0.0 si vide."""
    if not detections:
        return 0.0
    return sum(d["conf"] for d in detections) / len(detections)


@app.post("/solve-auto", response_model=SolveResponse)
async def solve_auto(
    file: UploadFile = File(..., description="Image (png/jpg/webp)"),
    model: Optional[str] = Query(default=None, description="Modèle YOLO (défaut: finetuned_v1)"),
    conf: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Seuil confiance YOLO"),
    imgsz: Optional[int] = Query(default=None, ge=256, le=2048, description="Taille d'entrée YOLO"),
    iou: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="IoU suppression doublons"),
    normalize: Optional[bool] = Query(default=None, description="Normaliser confusions (I/l/1, O/0)"),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Seuil cascade (fallback Gemini si avg conf < seuil)"),
    llm_model: Optional[str] = Query(default=None, description="Modèle Gemini pour fallback (ex: gemini-2.5-flash)"),
):
    """
    Cascade automatique : YOLO (finetuned) d'abord, Gemini en fallback
    si le texte est vide ou la confiance moyenne est trop basse.
    """
    # -- Résoudre le modèle YOLO --
    model_name = model or DEFAULT_CASCADE_MODEL
    if model_name not in engines:
        loaded = list(engines.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Modèle '{model_name}' non disponible. Chargés: {loaded}",
        )
    yolo_engine = engines[model_name]
    cascade_threshold = threshold if threshold is not None else DEFAULT_CASCADE_THRESHOLD

    # -- Valider l'upload --
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max allowed: {MAX_UPLOAD_BYTES} bytes.",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        meta = {
            "filename": file.filename,
            "content_type": file.content_type,
            "yolo_model": model_name,
            "cascade_threshold": cascade_threshold,
        }

        # ===== Phase 1 : YOLO =====
        t0 = time.perf_counter()
        yolo_error = False
        try:
            yolo_out = yolo_engine.solve(
                image_path=tmp_path,
                conf=conf,
                imgsz=imgsz,
                iou_threshold=iou,
                normalize_confusions=normalize,
                verbose=False,
            )
        except Exception as exc:
            yolo_error = True
            yolo_out = {"text": "", "detections": []}
            meta["yolo_error"] = str(exc)

        yolo_ms = (time.perf_counter() - t0) * 1000.0
        meta["yolo_elapsed_ms"] = round(yolo_ms, 2)

        yolo_text = yolo_out["text"]
        yolo_dets = yolo_out["detections"]
        avg_conf = _yolo_avg_confidence(yolo_dets)

        meta["yolo_text"] = yolo_text
        meta["yolo_avg_conf"] = round(avg_conf, 4)

        # ===== Décision : fallback nécessaire ? =====
        need_fallback = False
        fallback_reason = None

        if yolo_error:
            need_fallback = True
            fallback_reason = "yolo_error"
        elif not yolo_text:
            need_fallback = True
            fallback_reason = "empty_text"
        elif len(yolo_text) < DEFAULT_CASCADE_MIN_CHARS:
            need_fallback = True
            fallback_reason = "too_few_chars"
        elif avg_conf < cascade_threshold:
            need_fallback = True
            fallback_reason = "low_confidence"

        # ===== Phase 2 : Gemini fallback =====
        if need_fallback:
            gemini_available = bool(os.getenv(LLM_ENV_KEY))
            if not gemini_available:
                # Gemini non configuré → retourner le résultat YOLO avec warning
                meta["method"] = "yolo"
                meta["fallback"] = False
                meta["fallback_reason"] = fallback_reason
                meta["warning"] = f"Fallback nécessaire ({fallback_reason}) mais {LLM_ENV_KEY} non définie"
                meta["total_elapsed_ms"] = round(yolo_ms, 2)
                return SolveResponse(text=yolo_text, detections=yolo_dets, meta=meta)

            actual_llm_model = llm_model or LLM_DEFAULT_MODEL
            meta["gemini_model"] = actual_llm_model
            meta["fallback"] = True
            meta["fallback_reason"] = fallback_reason

            t1 = time.perf_counter()
            try:
                llm_engine = LlmOcrEngine(model=actual_llm_model)
                llm_out = llm_engine.solve(image_path=tmp_path)
                gemini_ms = (time.perf_counter() - t1) * 1000.0
                meta["gemini_elapsed_ms"] = round(gemini_ms, 2)
                meta["method"] = "yolo+gemini"
                meta["total_elapsed_ms"] = round(yolo_ms + gemini_ms, 2)
                return SolveResponse(
                    text=llm_out["text"],
                    detections=yolo_dets,
                    meta=meta,
                )
            except Exception as llm_exc:
                gemini_ms = (time.perf_counter() - t1) * 1000.0
                meta["gemini_elapsed_ms"] = round(gemini_ms, 2)
                meta["gemini_error"] = str(llm_exc)
                if yolo_text:
                    meta["method"] = "yolo"
                    meta["warning"] = "Gemini fallback échoué, résultat YOLO retourné"
                    meta["total_elapsed_ms"] = round(yolo_ms + gemini_ms, 2)
                    return SolveResponse(text=yolo_text, detections=yolo_dets, meta=meta)
                raise HTTPException(
                    status_code=502,
                    detail=f"YOLO et Gemini ont échoué. YOLO: {fallback_reason}, Gemini: {llm_exc}",
                )
        else:
            # YOLO a réussi avec confiance suffisante
            meta["method"] = "yolo"
            meta["fallback"] = False
            meta["total_elapsed_ms"] = round(yolo_ms, 2)
            return SolveResponse(text=yolo_text, detections=yolo_dets, meta=meta)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
