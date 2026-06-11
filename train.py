import csv
import os
from pathlib import Path

import torch
from ultralytics import YOLO

# Reduce GPU memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Split to train on — change this one line to switch splits ──────────────
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
# ─────────────────────────────────────────────────────────────────────────────

# ── Training hyperparameters ──────────────────────────────────────────────
MAX_EPOCHS     = 500    # safety ceiling — early stopping triggers well before this
PATIENCE       = 10     # stop if val metric doesn't improve for this many epochs
WARMUP_EPOCHS  = 3      # linear lr ramp at start (~500–1000 iters depending on split)
IMGSZ          = 640   # 416 was too small for DIOR's small objects (vehicles ~10px at 800→416); 640 keeps them ~8px
BATCH          = 4
LR0            = 1e-3   # initial lr (matches SSD)
MOMENTUM       = 0.9
WEIGHT_DECAY   = 5e-4
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent                  # ~/arman/YOLOv8
DATA_YAML    = BASE_DIR / 'data' / f'{SPLIT}.yaml'
RESULTS_DIR  = BASE_DIR / 'results'                   # ~/arman/YOLOv8/results
LOG_CSV      = RESULTS_DIR / SPLIT / 'loss_log.csv'
DEVICE       = '0' if torch.cuda.is_available() else 'cpu'


def make_csv_logger():
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Append if log already exists (resume run), write header only on first run
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, 'a', newline='') as f:
        if write_header:
            csv.writer(f).writerow([
                'epoch', 'train_box_loss', 'train_cls_loss', 'train_dfl_loss',
                'val_box_loss', 'val_cls_loss', 'val_dfl_loss',
                'mAP50', 'lr',
            ])

    def on_fit_epoch_end(trainer):
        m   = trainer.metrics
        lr  = trainer.optimizer.param_groups[0]['lr']
        row = [
            trainer.epoch + 1,
            f"{m.get('train/box_loss', 0):.4f}",
            f"{m.get('train/cls_loss', 0):.4f}",
            f"{m.get('train/dfl_loss', 0):.4f}",
            f"{m.get('val/box_loss',   0):.4f}",
            f"{m.get('val/cls_loss',   0):.4f}",
            f"{m.get('val/dfl_loss',   0):.4f}",
            f"{m.get('metrics/mAP50(B)', 0):.4f}",
            f"{lr:.6f}",
        ]
        with open(LOG_CSV, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    return on_fit_epoch_end


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Run convert_to_yolo.py first."
        )

    # Load YOLOv8n with ImageNet-1K pretrained backbone (backbone trained on ImageNet,
    # detection head fine-tuned on COCO — standard ultralytics transfer learning setup)
    model = YOLO('yolov8n.pt')
    model.add_callback('on_fit_epoch_end', make_csv_logger())

    model.train(
        data=str(DATA_YAML),
        epochs=MAX_EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        lrf=0.01,           # final lr = lr0 * lrf (ReduceLROnPlateau-like decay at end)
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=0.8,
        optimizer='SGD',
        pretrained=True,
        device=DEVICE,
        project=str(RESULTS_DIR),
        name=SPLIT,
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,
        save_period=50,     # periodic snapshots every 50 epochs (matches SSD)
    )

    print(f"\nTraining complete. Best checkpoint: {RESULTS_DIR / SPLIT / 'weights' / 'best.pt'}")
    print(f"Loss log: {LOG_CSV}")


if __name__ == '__main__':
    main()
