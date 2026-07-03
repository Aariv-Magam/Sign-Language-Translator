from pathlib import Path
import argparse
import os
import time
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'sign_language_rgb' / 'weights' / 'best.pt'

DEFAULT_CLASS_NAMES = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]


def get_class_names(model: YOLO | None = None) -> list[str]:
    if model is not None:
        names = getattr(model, 'names', None) or getattr(model.model, 'names', None)
        if names:
            return [str(names[i]) for i in sorted(names)]
    return DEFAULT_CLASS_NAMES


def resolve_window_mode(show_window: bool, env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    display = env.get('DISPLAY', '')
    return show_window and bool(display)


def update_translation_text(
    current_text: list[str],
    letter_buffer: list[str],
    detected_labels: list[str],
    last_label: str | None,
    last_change_time: float,
    now: float,
    debounce_seconds: float = 0.75,
) -> tuple[list[str], list[str], str | None, float]:
    if not detected_labels:
        if letter_buffer and now - last_change_time >= debounce_seconds:
            word = resolve_word_from_letters(letter_buffer)
            if word is not None:
                current_text.append(word)
            else:
                current_text.extend(letter_buffer)
            letter_buffer.clear()
        return current_text, letter_buffer, last_label, last_change_time

    label = detected_labels[0]
    if label == last_label:
        return current_text, letter_buffer, last_label, last_change_time

    letter_buffer.append(label)
    if len(letter_buffer) >= 2:
        candidate = resolve_word_from_letters(letter_buffer)
        if candidate is not None:
            current_text.append(candidate)
            letter_buffer.clear()
            return current_text, letter_buffer, label, now

    if now - last_change_time >= debounce_seconds and letter_buffer:
        if len(letter_buffer) > 1:
            current_text.extend(letter_buffer[:-1])
            letter_buffer = letter_buffer[-1:]
        else:
            current_text.append(letter_buffer[-1])
            letter_buffer.clear()

    return current_text, letter_buffer, label, now


WORD_DICTIONARY = {
    'HELLO': 'HELLO',
    'THANKYOU': 'THANK YOU',
    'PLEASE': 'PLEASE',
    'HELP': 'HELP',
    'SORRY': 'SORRY',
    'GOODBYE': 'GOODBYE',
    'YES': 'YES',
    'NO': 'NO',
    'MY': 'MY',
    'NAME': 'NAME',
    'IS': 'IS',
    'YOU': 'YOU',
    'ME': 'ME',
    'LOVE': 'LOVE',
    'SCHOOL': 'SCHOOL',
    'FAMILY': 'FAMILY',
    'FRIEND': 'FRIEND',
    'WATER': 'WATER',
    'FOOD': 'FOOD',
    'HOME': 'HOME',
    'WORK': 'WORK',
    'TIME': 'TIME',
    'TODAY': 'TODAY',
    'TOMORROW': 'TOMORROW',
    'WHERE': 'WHERE',
    'WHEN': 'WHEN',
    'WHY': 'WHY',
    'WHAT': 'WHAT',
    'WHO': 'WHO',
    'NEED': 'NEED',
    'WANT': 'WANT',
    'LIKE': 'LIKE',
    'LEARN': 'LEARN',
}


def resolve_word_from_letters(labels: list[str]) -> str | None:
    if not labels:
        return None
    candidate = ''.join(labels).upper()
    return WORD_DICTIONARY.get(candidate)


def format_translation_result(translated_text: list[str]) -> str:
    if translated_text:
        return f'Final translation: {" ".join(translated_text)}'
    return 'No signs were detected.'


def emit_final_translation(translated_text: list[str]) -> None:
    print(format_translation_result(translated_text), flush=True)


def invert_image(image: cv2.Mat) -> cv2.Mat:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def _detect_image(model: YOLO, image: cv2.Mat, conf: float, imgsz: int) -> list[str]:
    results = model(image, conf=conf, imgsz=imgsz, stream=False)[0]
    return [str(int(box.cls[0])) for box in results.boxes], results


def predict_image(image_path: str, conf: float = 0.25) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model weights not found at {MODEL_PATH}. Train the model first with python train_sign_model.py.'
        )

    model = YOLO(str(MODEL_PATH))
    class_names = get_class_names(model)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    labels = []
    results = None

    labels, results = _detect_image(model, image, conf, 320)
    if not labels:
        alt_image = invert_image(image)
        labels, results = _detect_image(model, alt_image, 0.01, 640)
        if labels:
            print('No detections on the original image; using inverted-image fallback.')

    if not labels:
        print('Detected symbols: []')
        print('English meaning: ') 
        return

    labels = [class_names[int(label)] if 0 <= int(label) < len(class_names) else label for label in labels]

    translated_words = []
    for label in labels:
        word = resolve_word_from_letters([label])
        translated_words.append(word if word is not None else label)

    print('Detected symbols:', labels)
    print('English meaning:', ' '.join(translated_words))


def predict_webcam(conf: float = 0.25, source: int = 0, max_frames: int | None = None, show_window: bool = True) -> None:
    effective_show_window = resolve_window_mode(show_window)
    if not effective_show_window:
        print('No display detected; running webcam inference without showing a window.')

    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        class_names = get_class_names(model)
    else:
        model = None
        class_names = DEFAULT_CLASS_NAMES
        print('No trained weights found yet. The webcam will start, but predictions will be limited until training finishes.')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam. Check camera permissions or try another camera index.')

    print('Webcam connected. Press q to quit, or Ctrl+C to stop and print the final translation.', flush=True)
    frame_count = 0
    translated_text: list[str] = []
    letter_buffer: list[str] = []
    last_label: str | None = None
    last_change_time = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if model is not None:
                results = model(frame, conf=conf, stream=False)[0]
                detected_labels = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                    detected_labels.append(cls_name)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cls_name, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                now = time.time()
                translated_text, letter_buffer, last_label, last_change_time = update_translation_text(
                    translated_text,
                    letter_buffer,
                    detected_labels,
                    last_label,
                    last_change_time,
                    now,
                )
            else:
                cv2.putText(frame, 'Model not trained yet', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, 'Run: python train_sign_model.py', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if effective_show_window:
                cv2.putText(frame, f'Translation: {" ".join(translated_text)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('ASL Translator', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    except KeyboardInterrupt:
        print('Stopping capture...', flush=True)
    finally:
        cap.release()
        if effective_show_window:
            cv2.destroyAllWindows()
        emit_final_translation(translated_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ASL sign translation with a trained YOLO model.')
    parser.add_argument('--image', type=str, help='Path to an image file to classify')
    parser.add_argument('--webcam', action='store_true', help='Run live webcam inference')
    parser.add_argument('--source', type=int, default=0, help='Webcam index to use (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--max-frames', type=int, default=None, help='Optional limit for webcam frames for testing')
    parser.add_argument('--no-window', action='store_true', help='Run webcam inference without showing the OpenCV window')
    args = parser.parse_args()

    if args.webcam or not args.image:
        predict_webcam(conf=args.conf, source=args.source, max_frames=args.max_frames, show_window=not args.no_window)
    elif args.image:
        predict_image(args.image, conf=args.conf)
