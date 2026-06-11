"""
Convert DIOR Pascal VOC XML annotations to YOLO format.
Run once before training — safe to re-run (overwrites existing files).

What this script produces
─────────────────────────
1. One .txt label file per trainval image, written NEXT TO the .jpg in
   JPEGImages-trainval/.  Ultralytics resolves label paths by replacing the
   image extension with .txt in the same directory when no /images/ component
   is present in the path, so this is the correct location.

2. Per-split image-list files under YOLOv8/data/:
     split_005_train.txt   ← absolute paths to the 5% training images
     split_005_val.txt     ← absolute paths to all val.txt images
     ... (same for 025, 050, 075, 100)

3. Per-split YAML config files under YOLOv8/data/:
     split_005.yaml, split_025.yaml, split_050.yaml,
     split_075.yaml, split_100.yaml

Split logic
───────────────────────────────────────────────────
  all_trainval_ids = train.txt IDs + val.txt IDs   (11,725 total)
  training subset  = [all_trainval_ids[i] for i in split_NNN_indices.npy
                       if i < len(train_ids)]       ← only train.txt images
  validation set   = all val.txt IDs               (5,863, fixed across all splits)

  The index filter (i < TRAIN_SIZE) is critical: generate_splits.py sampled
  from the full 11,725 pool, so without it ~50% of training images would be
  val images, causing data leakage.

After running this script, start training with:
  cd YOLOv8 && python train.py
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

# ── directory layout ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR      = PROJECT_ROOT / 'JPEGImages-trainval'
ANN_DIR      = PROJECT_ROOT / 'Annotations' / 'Horizontal Bounding Boxes'
MAIN_DIR     = PROJECT_ROOT / 'Main'
NPY_DIR      = PROJECT_ROOT / 'Python_datareader'
DATA_DIR     = Path(__file__).resolve().parent / 'data'

SPLITS = ['split_005', 'split_025', 'split_050', 'split_075', 'split_100']

# DIOR 20 classes — order must match utils.py and dior_dataset.py
CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
]
_CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_ids(txt_path):
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def _xml_to_yolo_lines(xml_path):
    """Parse one VOC XML file and return YOLO-format annotation lines.

    Reads image dimensions from the <size> tag so the conversion is correct
    even if any image is not the standard 800×800.

    Returns a list of strings: "<class_id> <cx> <cy> <w> <h>"
    Returns an empty list if the XML has no recognised objects.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    if size is None:
        raise ValueError(f"<size> tag missing in {xml_path} — XML may be malformed or truncated")
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in _CLASS_TO_IDX:
            continue
        bb   = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)

        # Skip degenerate boxes — ultralytics drops them silently during training
        # but leaving them in label files obscures data quality issues
        if xmax <= xmin or ymax <= ymin:
            print(f"  WARNING: degenerate box ({xmin},{ymin},{xmax},{ymax}) in {xml_path.name} — skipped")
            continue

        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w  = (xmax - xmin)     / img_w
        h  = (ymax - ymin)     / img_h

        # Clamp to [0, 1] — guards against tiny float violations in annotations
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        lines.append(f"{_CLASS_TO_IDX[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


# ── step 1: write label files ─────────────────────────────────────────────────

def write_all_labels(all_ids, force=False):
    """Write one .txt label file per image into JPEGImages-trainval/.

    All splits share the same label files so existing files are skipped by
    default.  Pass force=True to regenerate all labels from scratch (e.g.
    after correcting XML annotations).

    Args:
        all_ids: list of image ID strings
        force:   if True, overwrite existing .txt files; if False, skip them
    """
    written = skipped_existing = missing_xml = 0

    for img_id in all_ids:
        lbl_path = IMG_DIR / f'{img_id}.txt'

        if lbl_path.exists() and not force:
            skipped_existing += 1
            continue

        xml_path = ANN_DIR / f'{img_id}.xml'
        if not xml_path.exists():
            # Trainval images should always have XML — warn if one is missing
            print(f"  WARNING: no XML found for {img_id} — writing empty label file")
            lbl_path.write_text('')
            missing_xml += 1
            continue

        lines = _xml_to_yolo_lines(xml_path)
        lbl_path.write_text('\n'.join(lines) + '\n')
        written += 1

    print(f"  Labels written : {written}")
    print(f"  Already existed: {skipped_existing} (skipped)")
    if missing_xml:
        print(f"  No XML found   : {missing_xml} (empty label files written — check dataset)")


# ── step 2 & 3: per-split image lists and YAML files ─────────────────────────

def write_image_list(img_ids, out_path):
    """Write absolute image paths, one per line (with trailing newline)."""
    lines = [(IMG_DIR / f'{img_id}.jpg').as_posix() for img_id in img_ids]
    out_path.write_text('\n'.join(lines) + '\n')


def write_yaml(split_name, train_list_path, val_list_path):
    yaml_path = DATA_DIR / f'{split_name}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"# DIOR dataset — {split_name}\n")
        f.write(f"train: {train_list_path.as_posix()}\n")
        f.write(f"val:   {val_list_path.as_posix()}\n")
        f.write(f"nc:    {len(CLASSES)}\n")
        f.write("names:\n")
        for name in CLASSES:
            f.write(f"  - {name}\n")
    print(f"  Wrote {yaml_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Load all trainval IDs in the same order as _DIORBase / generate_splits.py
    train_ids  = _read_ids(MAIN_DIR / 'train.txt')   # 5,862 IDs
    val_ids    = _read_ids(MAIN_DIR / 'val.txt')     # 5,863 IDs
    all_ids    = train_ids + val_ids                  # 11,725 IDs (matches .npy index space)
    TRAIN_SIZE = len(train_ids)                       # 5,862 — indices below this are train-only

    # ── Step 1: write label files for all trainval images ─────────────────────
    print("Step 1 — writing YOLO label files to JPEGImages-trainval/ ...")
    write_all_labels(all_ids)

    # ── Steps 2 & 3: per-split image lists and YAML files ─────────────────────
    print("\nStep 2 & 3 — writing image lists and YAML configs ...")
    for split in SPLITS:
        print(f"\n  {split}")

        # Apply .npy index to the combined trainval pool, then keep only
        # indices that fall in the train.txt portion (< TRAIN_SIZE).
        # generate_splits.py sampled from the full 11,725 pool — without this
        # filter ~50% of selected images would be from val.txt, causing data
        # leakage (training and evaluating on the same images).
        indices      = np.load(NPY_DIR / f'{split}_indices.npy')
        train_subset = [all_ids[i] for i in indices if i < TRAIN_SIZE]

        train_list = DATA_DIR / f'{split}_train.txt'
        val_list   = DATA_DIR / f'{split}_val.txt'

        write_image_list(train_subset, train_list)
        write_image_list(val_ids,      val_list)
        print(f"  train: {len(train_subset):,} images  |  val: {len(val_ids):,} images")

        write_yaml(split, train_list, val_list)

    print("\nDone. Run  python train.py  to start training.")


if __name__ == '__main__':
    main()
