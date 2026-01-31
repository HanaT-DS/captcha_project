"""
schemas.py – Modèles Pydantic pour les requêtes/réponses de l'API

Définit les schémas :
- Detection      : un caractère détecté (char, confiance, bounding box)
- SolveResponse  : réponse de /solve et /solve-auto (texte, détections, meta)
- ConfigResponse : configuration courante du serveur
- ModelInfo      : informations sur un modèle YOLO
- ModelsResponse : liste des modèles disponibles
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any


class Detection(BaseModel):
    char: str
    conf: float
    xyxy: Tuple[float, float, float, float]


class SolveResponse(BaseModel):
    text: str
    detections: List[Detection] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    model_path: str
    data_yaml_path: str
    default_conf: float
    default_imgsz: int
    default_iou: float
    normalize_confusions: bool


class ModelInfo(BaseModel):
    name: str
    description: str
    imgsz: int
    conf: float
    loaded: bool


class ModelsResponse(BaseModel):
    default: str
    models: List[ModelInfo]
