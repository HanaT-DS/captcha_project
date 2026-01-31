"""
yolo_ocr.py – Moteur OCR basé sur YOLO (détection de caractères)

Charge un modèle YOLO (Ultralytics) pour détecter les caractères
individuels d'un CAPTCHA, puis reconstruit le texte en triant
les boxes de gauche à droite.

Fournit :
- YoloOcrEngine : classe d'inférence (solve, predict)
- MODEL_REGISTRY : registre des modèles disponibles (baseline, finetuned_v1)
- get_model_config(), list_models(), is_model_loaded() : helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


# ----------------------------
# Registre des modèles disponibles
# ----------------------------

MODELS_DIR = Path(__file__).resolve().parent / "weights"
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "description": "Baseline YOLOv8s (Roboflow synthétique)",
        "weights": MODELS_DIR / "best.pt",
        "data_yaml": DATASETS_DIR / "Yolo-Captcha-Test-7" / "data.yaml",
        "imgsz": 640,
        "conf": 0.25,
    },
    "finetuned_v1": {
        "description": "Fine-tuné sur CAPTCHAs cible (v1)",
        "weights": MODELS_DIR / "best_ft_ft_v1.pt",
        "data_yaml": DATASETS_DIR / "captcha_target_v1_yolo" / "data.yaml",
        "imgsz": 512,
        "conf": 0.25,
        "stretch_to": (512, 512),  # Roboflow: "Stretch to 512x512"
    },
}


def list_models() -> List[str]:
    """Retourne les noms des modèles disponibles dans le registre."""
    return list(MODEL_REGISTRY.keys())


def get_model_config(name: str) -> Dict[str, Any]:
    """Retourne la config d'un modèle par son nom."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Modèle '{name}' inconnu. Disponibles: {available}")
    return MODEL_REGISTRY[name]


# ----------------------------
# Types / structures de sortie
# ----------------------------

@dataclass
class DetectedChar:
    """Une détection de caractère."""
    char: str
    conf: float
    xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)


# ----------------------------
# Utils géométrie (comme notebook)
# ----------------------------

def iou_xyxy(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-Union entre deux boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# ----------------------------
# Normalisation des confusions (optionnel)
# ----------------------------

def normalize_confusable_chars(text: str) -> str:
    """
    Normalise des caractères visuellement confondus (désactivé par défaut).
    Direction: lowercase_preference (aligné sur le notebook baseline).
    - I -> l  (majuscule I -> minuscule L)
    - O -> 0  (majuscule O -> chiffre zéro)
    """
    mapping = {
        'I': 'l',  # Majuscule I -> minuscule L
        'O': '0',  # Majuscule O -> chiffre 0
    }
    return ''.join(mapping.get(c, c) for c in text)


# ----------------------------
# Chargement modèle
# ----------------------------

def load_model(model_path: str | Path) -> YOLO:
    """
    Charge un modèle Ultralytics YOLO depuis un fichier .pt.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    return YOLO(str(model_path))



def load_names_from_data_yaml(data_yaml_path: str | Path) -> List[str]:
    """
    Lit data.yaml (Roboflow/YOLO) et retourne la liste des classes `names`.
    """
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml introuvable: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", None)
    if not isinstance(names, list) or len(names) == 0:
        raise ValueError("Le champ `names` est absent ou invalide dans data.yaml")

    return names



# ----------------------------
# Preprocessing (reproduit le traitement Roboflow)
# ----------------------------

def preprocess_image(
    image_path: str | Path,
    stretch_to: Tuple[int, int] | None = None,
) -> str | np.ndarray:
    """
    Applique le même preprocessing que Roboflow a fait au dataset d'entraînement.

    Si stretch_to=(512,512), l'image est stretchée (ratio cassé) à 512×512
    exactement comme Roboflow "Stretch to 512x512".

    Retourne :
    - le chemin original si pas de preprocessing
    - un numpy array BGR (prêt pour YOLO) sinon
    """
    if stretch_to is None:
        return str(image_path)

    img = Image.open(image_path).convert("RGB")
    img = img.resize(stretch_to, Image.BILINEAR)
    # PIL est RGB, YOLO attend BGR → inverser les canaux
    return np.array(img)[:, :, ::-1]


# ----------------------------
# Prédiction + extraction des boxes
# ----------------------------

def predict_chars(
    model: YOLO,
    image_path: str | Path,
    names: List[str],
    conf: float = 0.25,
    imgsz: int = 640,
    stretch_to: Tuple[int, int] | None = None,
    verbose: bool = False,
) -> List[DetectedChar]:
    """
    Lance YOLO sur une image et retourne une liste de caractères détectés.
    - names : liste id -> char (depuis data.yaml)
    - stretch_to : si défini, stretch l'image avant inférence (ex: (512, 512))
    """
    source = preprocess_image(image_path, stretch_to=stretch_to)
    results = model.predict(source, conf=conf, imgsz=imgsz, verbose=verbose)
    r = results[0]


    detected: List[DetectedChar] = []

    if r.boxes is None or len(r.boxes) == 0:
        return detected

    for b in r.boxes:
        c = float(b.conf)
        cls_id = int(b.cls)
        x1, y1, x2, y2 = map(float, b.xyxy[0])

        detected.append(
            DetectedChar(
                char=names[cls_id],
                conf=c,
                xyxy=(x1, y1, x2, y2),
            )
        )

    return detected


# ----------------------------
# Décodage: boxes -> texte (1 ligne)
# ----------------------------

def decode_prediction(
    detections: List[DetectedChar],
    iou_threshold: float = 0.6,
    normalize_confusions: bool = False,
) -> Tuple[str, List[DetectedChar]]:
    """
    Reconstruit un texte (1 ligne) à partir des detections YOLO.
    Étapes :
    1) Supprime doublons (si deux boxes se chevauchent trop : garder la meilleure conf)
    2) Trie gauche -> droite par x_center
    3) Concatène les caractères
    4) Optionnel: normalisation confusions

    Returns:
        (text, kept) - texte reconstruit + détections gardées après NMS
    """
    if not detections:
        return "", []

    # 1) Suppression des doublons par IoU : greedy
    # On trie par confiance décroissante et on garde une box si elle ne chevauche pas trop une déjà gardée.
    det_sorted = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: List[DetectedChar] = []

    for d in det_sorted:
        overlap = False
        for k in kept:
            if iou_xyxy(d.xyxy, k.xyxy) >= iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(d)

    # 2) Tri gauche -> droite
    def x_center(det: DetectedChar) -> float:
        x1, _, x2, _ = det.xyxy
        return (x1 + x2) / 2.0

    kept.sort(key=x_center)

    # 3) Concaténer
    text = "".join(d.char for d in kept)

    # 4) Normalisation optionnelle
    if normalize_confusions:
        text = normalize_confusable_chars(text)

    return text, kept


# ----------------------------
# Pipeline complet: predict + decode
# ----------------------------

def predict_and_decode(
    model: YOLO,
    image_path: str | Path,
    names: List[str],
    conf: float = 0.25,
    imgsz: int = 640,
    iou_threshold: float = 0.6,
    normalize_confusions: bool = False,
    stretch_to: Tuple[int, int] | None = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Pipeline complet.
    Retourne un dict avec :
    - text : texte reconstruit
    - detections : liste des detections (char/conf/box)
    """
    dets = predict_chars(model, image_path, names, conf=conf, imgsz=imgsz, stretch_to=stretch_to, verbose=verbose)
    text, kept = decode_prediction(dets, iou_threshold=iou_threshold, normalize_confusions=normalize_confusions)

    return {
        "text": text,
        "detections": [
            {"char": d.char, "conf": d.conf, "xyxy": d.xyxy}
            for d in kept
        ],
        "image_path": str(image_path),
        "conf": conf,
        "imgsz": imgsz,
    }

# ============================================================
# Classe "YoloOcrEngine" (wrapper propre autour des fonctions)
# ============================================================

class YoloOcrEngine:
    """
    Moteur OCR YOLO :
    - charge le modèle une seule fois
    - charge les classes (names) une seule fois
    - expose une méthode solve(image) -> dict résultat
    """

    def __init__(
        self,
        model_path: str | Path,
        data_yaml_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        iou_threshold: float = 0.6,
        normalize_confusions: bool = False,
        stretch_to: Tuple[int, int] | None = None,
        verbose: bool = False,
    ):
        # Chemins
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path)

        # Paramètres par défaut (tu peux les override dans solve() si tu veux)
        self.conf = conf
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.normalize_confusions = normalize_confusions
        self.stretch_to = stretch_to
        self.verbose = verbose

        # Charger modèle + classes une seule fois
        self.model = load_model(self.model_path)
        self.names = load_names_from_data_yaml(self.data_yaml_path)


    def solve(
        self,
        image_path: str | Path,
        conf: float | None = None,
        imgsz: int | None = None,
        iou_threshold: float | None = None,
        normalize_confusions: bool | None = None,
        verbose: bool | None = None,
    ) -> Dict[str, Any]:
        """
        Résout une image (OCR).
        Si tu ne passes pas un paramètre, on utilise celui par défaut de l'objet.
        """
        return predict_and_decode(
            model=self.model,
            image_path=image_path,
            names=self.names,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz if imgsz is None else imgsz,
            iou_threshold=self.iou_threshold if iou_threshold is None else iou_threshold,
            normalize_confusions=self.normalize_confusions if normalize_confusions is None else normalize_confusions,
            stretch_to=self.stretch_to,
            verbose=self.verbose if verbose is None else verbose,
        )
