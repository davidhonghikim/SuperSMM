"""
musicxml_utils.py
Utility to convert recognized symbol lists to a basic music21 Score for export.
"""

from typing import List, Dict, Any

try:
    import music21
except ImportError:
    music21 = None

import re


def parse_label(label: str):
    """
    Parse a label like 'gracenote-G4_quarter' into pitch and duration.
    Returns (pitch, duration, is_rest).
    """
    label = label.lower()
    # Example patterns: 'gracenote-g4_quarter', 'rest_eighth', 'g#5_sixteenth', etc.
    rest_match = re.search(r"rest", label)
    lyric = None
    # Try to find pitch (e.g., G4, G#4, Gb4, etc.)
    pitch_match = re.search(r"([a-g][b#]?\d)", label)
    pitch = pitch_match.group(1).upper() if pitch_match else None
    # Try to find duration (quarter, eighth, etc.)
    duration_match = re.search(
        r"(whole|half|quarter|eighth|sixteenth|thirty_second)", label
    )
    duration = duration_match.group(1) if duration_match else "quarter"
    is_rest = bool(rest_match)
    return pitch, duration, is_rest


def symbols_to_score(symbols: List[Dict[str, Any]], title: str = None):
    """
    Convert a list of recognized symbols into a music21.stream.Score with correct notes/rests and durations.
    Args:
        symbols: List of recognized symbol dicts (with 'label' key)
        title: Optional title for the score
    Returns:
        music21.stream.Score
    """
    if music21 is None:
        raise ImportError("music21 is required for MusicXML export.")
    score = music21.stream.Score()
    if title:
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = title
    part = music21.stream.Part()
    for symbol in symbols:
        label = symbol.get("label", "")
        pitch, duration, is_rest = parse_label(label)
        # Add lyric if present
        lyric = symbol.get("lyric", None)
        if is_rest:
            n = music21.note.Rest(type=duration)
        elif pitch:
            n = music21.note.Note(pitch, type=duration)
        else:
            # Fallback: treat as C4 quarter note
            n = music21.note.Note("C4", type="quarter")
        if lyric:
            n.lyric = lyric
        part.append(n)
    score.append(part)
    return score
