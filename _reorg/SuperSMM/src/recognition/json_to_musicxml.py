import json
from pathlib import Path
from music21 import stream, note, clef, meter, key, converter


def label_to_music21(label):
    # Placeholder: Map OMR label to music21 object
    # Extend this mapping with your actual symbol vocabulary
    if label.startswith("note"):
        # Example: 'note_C4_quarter' -> pitch C4, quarter duration
        parts = label.split("_")
        pitch = parts[1] if len(parts) > 1 else "C4"
        dur = parts[2] if len(parts) > 2 else "quarter"
        return note.Note(pitch, type=dur)
    elif label == "clef_G":
        return clef.TrebleClef()
    elif label == "clef_F":
        return clef.BassClef()
    elif label == "meter_4/4":
        return meter.TimeSignature("4/4")
    elif label == "key_C":
        return key.Key("C")
    elif label == "rest_quarter":
        return note.Rest(type="quarter")
    # Extend for more symbols
    else:
        return None


def json_to_musicxml(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    symbols = data.get("symbols", [])
    # Sort by x position (left to right)
    symbols = sorted(symbols, key=lambda s: s["position"]["x"])
    s = stream.Stream()
    for sym in symbols:
        label = sym.get("label", "unknown")
        obj = label_to_music21(label)
        if obj:
            s.append(obj)
    # Write to MusicXML
    s.write("musicxml", fp=str(output_path))
    print(f"[INFO] Exported MusicXML to {output_path}")


if __name__ == "__main__":
    # Example usage: convert a processed symbol JSON to MusicXML
    project_root = Path(__file__).resolve().parent.parent.parent
    json_path = (
        project_root / "data/output/symbol_recognition/processed/page_1/symbols.json"
    )
    output_path = project_root / "data/output/musicxml/page_1.musicxml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_to_musicxml(json_path, output_path)
