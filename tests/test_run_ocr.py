"""
test_run_ocr.py – Test unitaire YOLO : résout un CAPTCHA via YoloOcrEngine

Charge un modèle YOLO (baseline ou finetuned) et prédit le texte
d'une image CAPTCHA donnée. Affiche le texte résolu et le nombre
de détections (boxes de caractères).

Usage:
    uv run python -m tests.test_run_ocr --model baseline --image "path/to/captcha.png"
    uv run python -m tests.test_run_ocr --model finetuned_v1 --image "path/to/captcha.png"
"""

import argparse
from pathlib import Path

from models.yolo_ocr import YoloOcrEngine, MODEL_REGISTRY, get_model_config, list_models



def main():
    available = ", ".join(list_models())

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Chemin vers une image")
    parser.add_argument(
        "--model",
        default="baseline",
        help=f"Nom du modèle ({available}) ou chemin vers un .pt",
    )
    parser.add_argument("--data-yaml", default=None, help="Chemin vers data.yaml (auto si modèle du registre)")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    # Vérifs chemins
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image introuvable: {args.image}")

    # Résoudre le modèle : par nom (registre) ou par chemin (.pt)
    if args.model in MODEL_REGISTRY:
        cfg = get_model_config(args.model)
        model_path = cfg["weights"]
        data_yaml = args.data_yaml or cfg["data_yaml"]
        conf = args.conf if args.conf is not None else cfg["conf"]
        imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
        print(f"Modèle: {args.model} ({cfg['description']})")
    else:
        model_path = args.model
        data_yaml = args.data_yaml or "datasets/Yolo-Captcha-Test-7/data.yaml"
        conf = args.conf if args.conf is not None else 0.25
        imgsz = args.imgsz if args.imgsz is not None else 640
        print(f"Modèle: {model_path} (chemin custom)")

    engine = YoloOcrEngine(model_path, data_yaml)

    out = engine.solve(
        image_path=args.image,
        conf=conf,
        imgsz=imgsz,
        iou_threshold=args.iou,
        normalize_confusions=args.normalize,
    )

    print(f"Texte : {out['text']}")
    print(f"Nb detections : {len(out['detections'])}")


if __name__ == "__main__":
    main()

# Exemples:
# uv run python -m tests.test_run_ocr --model baseline --image "datasets/Yolo-Captcha-Test-7/test/images/0_jpg.rf.273902948cba28c1539602776b62468c.jpg"
# uv run python -m tests.test_run_ocr --model finetuned_v1 --image "path/to/captcha.png"
# uv run python -m tests.test_run_ocr --model "models/weights/custom.pt" --data-yaml "datasets/custom/data.yaml" --image "img.png"
