from pathlib import Path
import shutil
import zipfile
import os
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_ZIP = Path.home() / 'Downloads' / 'asl-sign-language-dataset-rgb-hsv-and-grayscale.zip'
EXTRACT_DIR = ROOT / 'data' / 'rgb_dataset'
DATA_YAML = EXTRACT_DIR / 'data.yaml'


def extract_dataset(zip_path: Path, out_dir: Path) -> None:
    if (out_dir / 'train' / 'images').exists() and (out_dir / 'train' / 'labels').exists():
        print(f'Using existing extracted dataset at {out_dir}')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        source_targets = [
            ('RGB/rgb/train/images', out_dir / 'train' / 'images'),
            ('RGB/rgb/train/labels', out_dir / 'train' / 'labels'),
            ('RGB/rgb/val/images', out_dir / 'val' / 'images'),
            ('RGB/rgb/val/labels', out_dir / 'val' / 'labels'),
        ]
        for source_prefix, target_dir in source_targets:
            target_dir.mkdir(parents=True, exist_ok=True)
            for info in zf.infolist():
                if not info.filename.endswith('/') and info.filename.startswith(source_prefix + '/'):
                    rel_name = Path(info.filename).name
                    target_path = target_dir / rel_name
                    with zf.open(info) as src, open(target_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)

    print(f'Extracted dataset to {out_dir}')


def write_data_yaml(out_dir: Path) -> None:
    names = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
    yaml_text = f"""train: {out_dir / 'train' / 'images'}
val: {out_dir / 'val' / 'images'}

nc: {len(names)}
names: {names}
"""
    DATA_YAML.write_text(yaml_text)
    print(f'Wrote dataset config to {DATA_YAML}')


def main() -> None:
    if not DATA_ZIP.exists():
        raise FileNotFoundError(f'Dataset archive not found: {DATA_ZIP}')

    extract_dataset(DATA_ZIP, EXTRACT_DIR)
    write_data_yaml(EXTRACT_DIR)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    model = YOLO('yolov8n.pt')
    model.train(
        data=str(DATA_YAML),
        epochs=3,
        imgsz=320,
        batch=16,
        device=device,
        project=str(ROOT / 'models'),
        name='sign_language_rgb',
        exist_ok=True,
        workers=4,
        patience=0,
    )

    print('Training complete. Best weights saved to:', ROOT / 'models' / 'sign_language_rgb' / 'weights' / 'best.pt')


if __name__ == '__main__':
    main()
