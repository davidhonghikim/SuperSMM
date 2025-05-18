"""
music_exporter.py
Handles exporting recognized music data to MusicXML and MIDI formats.
"""

from pathlib import Path
from typing import Any
import logging

try:
    import music21  # For MusicXML and MIDI export
except ImportError:
    music21 = None
    logging.warning(
        "music21 library not found. MusicXML and MIDI export will not work."
    )


def export_musicxml(score_data: Any, output_path: Path, title: str = None) -> Path:
    """
    Export the recognized score data to MusicXML format using music21.
    Args:
        score_data: music21.stream.Score or compatible object
        output_path: Path to save the MusicXML file
        title: Optional title to set on the score
    Returns:
        Path to the saved MusicXML file
    """
    if music21 is None:
        raise ImportError("music21 is required for MusicXML export.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Set title if provided and score_data supports it
    if title and hasattr(score_data, "metadata"):
        if score_data.metadata is None:
            score_data.metadata = music21.metadata.Metadata()
        score_data.metadata.title = title
    score_data.write("musicxml", fp=str(output_path))
    logging.info(f"MusicXML exported to {output_path}")
    return output_path


def export_midi(score_data: Any, output_path: Path) -> Path:
    """
    Export the recognized score data to MIDI format using music21.
    Args:
        score_data: music21.stream.Score or compatible object
        output_path: Path to save the MIDI file
    Returns:
        Path to the saved MIDI file
    """
    if music21 is None:
        raise ImportError("music21 is required for MIDI export.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_data.write("midi", fp=str(output_path))
    logging.info(f"MIDI exported to {output_path}")
    return output_path
