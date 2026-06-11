"""
Run object detection on a single image or directory using the best checkpoint.
Displays results with bounding boxes and saves them to the results folder.
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# ── Set split, source, and checkpoint ─────────────────────────────────────
SPLIT  = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
EPOCH  = None         # None = best checkpoint; integer = specific save_period snapshot
SOURCE = str(Path(__file__).resolve().parent.parent / 'JPEGImages-test' / '00001.jpg')
# ─────────────────────────────────────────────────────────────────────────────

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

BASE_DIR    = Path(__file__).parent                          # ~/arman/YOLOv8
WEIGHTS_DIR = BASE_DIR / 'results' / SPLIT / 'weights'
SAVE_DIR    = BASE_DIR / 'results' / SPLIT / 'detections'
DEVICE      = '0' if torch.cuda.is_available() else 'cpu'


def resolve_checkpoint():
    if EPOCH is None:
        ckpt = WEIGHTS_DIR / 'best.pt'
    else:
        ckpt = WEIGHTS_DIR / f'epoch{EPOCH}.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def main():
    ckpt = resolve_checkpoint()
    print(f"Model:  {ckpt}")
    print(f"Source: {SOURCE}")

    model   = YOLO(str(ckpt))
    results = model.predict(
        source=SOURCE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        device=DEVICE,
        save=True,
        save_txt=False,
        project=str(SAVE_DIR.parent),
        name='detections',
        exist_ok=True,
        verbose=True,
    )

    print(f"\nResults saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
