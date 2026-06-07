"""
YOLOv8n Convergence Study — LVIS Greedy-50 (50 classes, from scratch)

Workflow
--------
1. convert_to_yolo.py  — one-time: convert COCO annotations -> YOLO txt labels + data YAMLs
2. train.py            — set SPLIT at the top, then run to train from scratch
3. eval.py             — set SPLIT + EPOCH (None = best), evaluate mAP and per-class AP
4. detect.py           — set SPLIT + SOURCE, run detection on images

Usage
-----
    python convert_to_yolo.py       # run once before anything else
    python train.py                 # trains split_005 by default
    python eval.py
    python detect.py
"""
