"""
02_auto_annotate.py – Pré-annotation automatique des CAPTCHAs avec le modèle baseline.

Utilise YoloOcrEngine pour prédire le texte + boxes sur chaque image,
puis génère :
  - predictions.json  (détails complets)
  - predictions.csv   (vérification rapide)

Usage:
    uv run python -m scripts.data_prep.02_auto_annotate
    uv run python -m scripts.data_prep.02_auto_annotate --conf 0.15 --imgsz 640
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from models.yolo_ocr import YoloOcrEngine


# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_RAW_DIR = "datasets/captcha_target_v1/raw"
DEFAULT_OUT_DIR = "datasets/captcha_target_v1/annotations"
DEFAULT_MODEL = "models/weights/best.pt"
DEFAULT_DATA_YAML = "datasets/Yolo-Captcha-Test-7/data.yaml"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 960
DEFAULT_IOU = 0.6


# ── Annotation ────────────────────────────────────────────────────────

def auto_annotate(
    raw_dir: str = DEFAULT_RAW_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    model_path: str = DEFAULT_MODEL,
    data_yaml: str = DEFAULT_DATA_YAML,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    iou: float = DEFAULT_IOU,
) -> Path:
    """Pré-annote toutes les images de raw_dir avec le modèle baseline."""
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Charger le modèle
    engine = YoloOcrEngine(
        model_path=model_path,
        data_yaml_path=data_yaml,
        conf=conf,
        imgsz=imgsz,
        iou_threshold=iou,
    )

    # Lister les images (png/jpg/jpeg/webp)
    images = sorted(
        f for f in raw.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    if not images:
        print(f"Aucune image trouvée dans {raw.resolve()}")
        return out

    print(f"Images à annoter : {len(images)}")
    print(f"Modèle           : {model_path}")
    print(f"conf={conf}, imgsz={imgsz}, iou={iou}\n")

    records = []
    total_conf = 0.0
    total_boxes = 0

    for img_path in tqdm(images, desc="Annotation", unit="img"):
        try:
            result = engine.solve(
                image_path=str(img_path),
                conf=conf,
                imgsz=imgsz,
            )
        except Exception as e:
            tqdm.write(f"Erreur sur {img_path.name}: {e}")
            records.append({
                "image": img_path.name,
                "predicted_text": "",
                "verified_text": "",
                "confidence_avg": 0.0,
                "boxes": [],
                "error": str(e),
            })
            continue

        boxes = [
            {
                "char": d["char"],
                "box": [round(v, 1) for v in d["xyxy"]],
                "conf": round(d["conf"], 4),
            }
            for d in result["detections"]
        ]

        avg_conf = 0.0
        if boxes:
            avg_conf = sum(b["conf"] for b in boxes) / len(boxes)
            total_conf += avg_conf
            total_boxes += len(boxes)

        records.append({
            "image": img_path.name,
            "predicted_text": result["text"],
            "verified_text": "",
            "confidence_avg": round(avg_conf, 4),
            "boxes": boxes,
        })

    # ── JSON ──────────────────────────────────────────────────────
    json_path = out / "predictions.json"
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = out / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted_text", "verified_text"])
        for r in records:
            writer.writerow([r["image"], r["predicted_text"], r["verified_text"]])

    # ── Résumé ────────────────────────────────────────────────────
    n = len(records)
    global_avg = (total_conf / n) if n else 0.0

    print(f"\nImages traitées     : {n}")
    print(f"Confiance moyenne   : {global_avg:.4f}")
    print(f"Boxes totales       : {total_boxes}")
    print(f"JSON                : {json_path.resolve()}")
    print(f"CSV                 : {csv_path.resolve()}")

    return out


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pré-annotation automatique des CAPTCHAs avec le modèle baseline"
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR,
        help=f"Dossier des images brutes (défaut: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Dossier de sortie annotations (défaut: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Chemin vers best.pt (défaut: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data-yaml", default=DEFAULT_DATA_YAML,
        help=f"Chemin vers data.yaml (défaut: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Seuil de confiance (défaut: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"Taille d'entrée YOLO (défaut: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"Seuil IoU suppression doublons (défaut: {DEFAULT_IOU})",
    )
    args = parser.parse_args()

    auto_annotate(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        model_path=args.model,
        data_yaml=args.data_yaml,
        conf=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()
