import json
from pathlib import Path
import cv2
import numpy as np
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from src.recognition.symbol_recognizer import SymbolRecognizer

# Set model and vocab paths (semantic model by default)
MODEL_PATH = "ml/models/resources/tf-deep-omr/Data/Models/symbol_recognition.h5"
VOCAB_PATH = "ml/models/resources/tf-deep-omr/Data/vocabulary_semantic.txt"


def recognize_and_update(symbols_dir, json_path):
    # Load symbol images
    symbol_imgs = []
    img_files = []
    for img_file in sorted(Path(symbols_dir).glob("symbol_*.png")):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            symbol_imgs.append(img)
            img_files.append(img_file)
    if not symbol_imgs:
        print(f"[WARN] No symbol images found in {symbols_dir}")
        return

    # Initialize recognizer
    recognizer = SymbolRecognizer(
        {
            "model_path": MODEL_PATH,
            "vocab_path": VOCAB_PATH,
            "confidence_threshold": 0.0,
        }
    )
    # Recognize symbols
    results = recognizer.recognize_symbols_batch(symbol_imgs)
    # Print top-3 predictions for first 10 symbols
    print("\n[DEBUG] Top-3 predictions for first 10 symbols:")
    for i, res in enumerate(results[:10]):
        print(f"Symbol {i}:")
        if "alternatives" in res and isinstance(res["alternatives"], list):
            preds = [
                {"label": res.get("label"), "confidence": res.get("confidence")}
            ] + res["alternatives"][:2]
            for j, pred in enumerate(preds):
                print(
                    f"  Top {j+1}: {pred.get('label')} (conf: {pred.get('confidence')})"
                )
        else:
            print(f"  Only: {res.get('label')} (conf: {res.get('confidence')})")

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    symbols = data.get("symbols", [])

    # Update JSON with recognition results
    for i, res in enumerate(results):
        if i < len(symbols):
            symbols[i]["label"] = res.get("label", "unknown")
            symbols[i]["confidence"] = res.get("confidence", 0.0)
            symbols[i]["class_index"] = res.get("class_index")
            symbols[i]["alternatives"] = res.get("alternatives", [])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Updated {json_path} with recognized labels.")


if __name__ == "__main__":
    # Example for page_1
    project_root = Path(__file__).resolve().parent.parent.parent
    page = "page_1"
    symbols_dir = (
        project_root / f"data/output/symbol_recognition/processed/{page}/symbols"
    )
    json_path = (
        project_root / f"data/output/symbol_recognition/processed/{page}/symbols.json"
    )
    recognize_and_update(symbols_dir, json_path)
