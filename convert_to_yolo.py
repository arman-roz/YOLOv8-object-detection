"""
Convert COCO-format split annotations to YOLO format.
Creates label .txt files next to images and data/<split>.yaml files.
Run once before training — safe to re-run (overwrites existing labels).

Output structure (added alongside existing images):
  splits/split_005/train/labels/000000000149.txt
  splits/split_005/val/labels/000000000149.txt
  data/split_005.yaml
  ...
"""

import json
from collections import defaultdict
from pathlib import Path

SPLITS_DIR = Path(__file__).parent.parent / 'Dataset_1' / 'splits'
DATA_DIR   = Path(__file__).parent / 'data'
SPLITS     = ['split_005', 'split_025', 'split_050', 'split_075', 'split_100']


def build_category_map(categories):
    """LVIS category_id -> YOLO class index (sorted by category_id for reproducibility)."""
    sorted_cats = sorted(categories, key=lambda c: c['id'])
    id_to_idx   = {cat['id']: idx for idx, cat in enumerate(sorted_cats)}
    names        = [cat['name'] for cat in sorted_cats]
    return id_to_idx, names


def convert_subset(split_name, subset, cat_id_to_idx):
    ann_path = SPLITS_DIR / split_name / subset / 'annotations.json'
    lbl_dir  = SPLITS_DIR / split_name / subset / 'labels'
    lbl_dir.mkdir(exist_ok=True)

    with open(ann_path) as f:
        data = json.load(f)

    img_info   = {img['id']: img for img in data['images']}
    ann_by_img = defaultdict(list)
    for ann in data['annotations']:
        ann_by_img[ann['image_id']].append(ann)

    skipped = 0
    for img in data['images']:
        img_id = img['id']
        img_w  = img['width']
        img_h  = img['height']

        lines = []
        for ann in ann_by_img.get(img_id, []):
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_idx:
                skipped += 1
                continue
            cls_idx = cat_id_to_idx[cat_id]
            x, y, w, h = ann['bbox']  # COCO: top-left x,y + width,height
            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h
            # Clamp to [0, 1] — small violations can occur from float COCO coords
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w_n = max(0.0, min(1.0, w_n))
            h_n = max(0.0, min(1.0, h_n))
            lines.append(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        lbl_file = lbl_dir / f"{img_id:012d}.txt"
        with open(lbl_file, 'w') as f:
            f.write('\n'.join(lines))

    if skipped:
        print(f"  [{split_name}/{subset}] {skipped} annotations skipped (unknown category_id)")


def write_yaml(split_name, class_names):
    train_imgs = (SPLITS_DIR / split_name / 'train' / 'images').as_posix()
    val_imgs   = (SPLITS_DIR / split_name / 'val'   / 'images').as_posix()
    yaml_path  = DATA_DIR / f'{split_name}.yaml'

    with open(yaml_path, 'w') as f:
        f.write(f"train: {train_imgs}\n")
        f.write(f"val:   {val_imgs}\n")
        f.write(f"nc:    {len(class_names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")

    print(f"  Wrote {yaml_path}")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Derive category map from split_005 (all splits share the same 50 categories)
    ref_ann = SPLITS_DIR / 'split_005' / 'train' / 'annotations.json'
    with open(ref_ann) as f:
        ref_data = json.load(f)
    cat_id_to_idx, class_names = build_category_map(ref_data['categories'])

    print(f"Categories ({len(class_names)}):")
    for idx, name in enumerate(class_names):
        print(f"  {idx:2d}: {name}")

    for split in SPLITS:
        print(f"\n{split}")
        for subset in ['train', 'val']:
            print(f"  converting {subset}...")
            convert_subset(split, subset, cat_id_to_idx)
        write_yaml(split, class_names)

    print("\nDone. Run train.py to start training.")


if __name__ == '__main__':
    main()
