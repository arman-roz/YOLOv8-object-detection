"""
Evaluate best (or a specific epoch) checkpoint on the validation set.
Prints per-class AP and overall mAP@0.50 / mAP@0.50:0.95.
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# ── Set split and checkpoint ───────────────────────────────────────────────
SPLIT  = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
EPOCH  = None         # None = best checkpoint; integer = specific save_period snapshot
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent                          # ~/arman/YOLOv8
DATA_YAML   = BASE_DIR / 'data' / f'{SPLIT}.yaml'
WEIGHTS_DIR = BASE_DIR / 'results' / SPLIT / 'weights'
DEVICE      = '0' if torch.cuda.is_available() else 'cpu'


def resolve_checkpoint():
    if EPOCH is None:
        ckpt = WEIGHTS_DIR / 'best.pt'
    else:
        # ultralytics saves periodic checkpoints as epoch{N}.pt
        ckpt = WEIGHTS_DIR / f'epoch{EPOCH}.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def main():
    ckpt = resolve_checkpoint()
    print(f"Evaluating: {ckpt}")

    model   = YOLO(str(ckpt))
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,   # must match IMGSZ in train.py
        batch=8,
        device=DEVICE,
        verbose=True,
        plots=True,
        save_json=True,
    )

    print(f"\nmAP@0.50:       {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95:  {metrics.box.map:.4f}")
    print(f"Precision:      {metrics.box.mp:.4f}")
    print(f"Recall:         {metrics.box.mr:.4f}")

    # Per-class AP
    print("\nPer-class AP@0.50:")
    names = model.names
    for idx, ap in enumerate(metrics.box.ap50):
        print(f"  {names[idx]:<30s} {ap:.4f}")


if __name__ == '__main__':
    main()
