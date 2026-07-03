# Sign Language Translatorq

This project trains a YOLOv8-based detector for American Sign Language (ASL) signs from the downloaded dataset archive.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended
- Internet access for installing dependencies

## Install dependencies

```bash
cd /home/nvidia/Sign-Language-Translator
pip install ultralytics opencv-python
```

## Train the model

```bash
python train_sign_model.py
```

This will:
- extract the RGB portion of the downloaded Kaggle archive,
- create a dataset configuration,
- train a YOLOv8 model for 3 epochs,
- save the best weights to models/sign_language_rgb/weights/best.pt.

## Run inference on an image

```bash
python translate_sign.py --image /path/to/your/image.jpg
```

Example output:

```text
Detected symbols: ['A', 'B']
Translated text: AB
```

## Run inference with the webcam

```bash
python translate_sign.py --webcam
```

Press q to stop the webcam window.
Video link: file:///C:/Users/Student/AppData/Local/CapCut/Videos/NVIDIA%20Submission.mp4

