import os
import json
import logging
import re
import time
import cv2

from typing import Dict, List, Any, Optional, Union
import xml.etree.ElementTree as ET
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# Music theory constants
PITCH_CLASSES = ["C", "D", "E", "F", "G", "A", "B"]
OCTAVE_RANGE = range(1, 7)
NOTE_DURATIONS = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "sixteenth": 0.25}

# Ensure debug logging directory exists
DEBUG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "debug_logs")
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)


class MusicXMLConverter:
    def __init__(self, output_dir="exports"):
        """
        Convert OMR processing results to MusicXML

        Args:
            output_dir (str): Directory to save exported MusicXML files
        """
        self.logger = logging.getLogger("musicxml_converter")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def convert(
        self, omr_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Convert OMR processing results to MusicXML

        Args:
            omr_results (Dict[str, Any]): Results from OMR processing

        Returns:
            str: Path to generated MusicXML file
        """
        # Validate input
        if not isinstance(omr_results, dict):
            raise ValueError("Input must be a dictionary containing OMR results")

        try:
            # Debug: Save raw OMR results
            raw_results_path = os.path.join(DEBUG_OUTPUT_DIR, "raw_omr_results.json")
            with open(raw_results_path, "w") as f:
                json.dump(omr_results, f, indent=2, cls=NumpyEncoder)
            self.logger.info(f"Raw OMR results saved to {raw_results_path}")

            # Create XML structure
            score_partwise = ET.Element("score-partwise", version="4.0")

            # Add work title
            work = ET.SubElement(score_partwise, "work")
            work_title = omr_results.get("metadata", {}).get("title", "Untitled")
            ET.SubElement(work, "work-title").text = work_title

            # Add identification
            identification = ET.SubElement(score_partwise, "identification")
            creator = ET.SubElement(identification, "creator", type="transcription")
            creator.text = "SuperSMM Optical Music Recognition"

            # Add part-list
            part_list = ET.SubElement(score_partwise, "part-list")
            score_part = ET.SubElement(part_list, "score-part", id="P1")
            ET.SubElement(score_part, "part-name").text = "Sheet Music"

            # Add part
            part = ET.SubElement(score_partwise, "part", id="P1")

            # Prepare debug logs for symbols
            symbols_debug_log = []

            # Normalize input format
            symbols = omr_results.get("symbols", [])
            metadata = omr_results.get("metadata", {})

            # Validate symbols
            if not symbols:
                self.logger.warning("No symbols found in OMR results")

            # Create measure
            measure = ET.SubElement(part, "measure", number="1")

            # Pitches for cycling
            pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]
            page_symbols = []

            # Process symbols
            for idx, symbol in enumerate(symbols):
                try:
                    # Ensure symbol is a dictionary
                    if not isinstance(symbol, dict):
                        symbol = {"label": str(symbol)}

                    symbol_label = symbol.get("label", f"Unknown Symbol {idx}")
                    note = self._create_note_element(
                        symbol, pitch=pitches[idx % len(pitches)]
                    )
                    measure.append(note)

                    # Log symbol details
                    page_symbols.append(
                        {
                            "index": idx,
                            "label": symbol_label,
                            "pitch": pitches[idx % len(pitches)],
                        }
                    )
                except Exception as symbol_error:
                    self.logger.error(f"Error processing symbol {idx}: {symbol_error}")

            # Collect symbols for debug logging
            symbols_debug_log.append(page_symbols)

            # Debug: Save symbol processing details
            symbols_log_path = os.path.join(
                DEBUG_OUTPUT_DIR, "symbol_processing_log.json"
            )
            with open(symbols_log_path, "w") as f:
                json.dump(symbols_debug_log, f, indent=2, cls=NumpyEncoder)
            self.logger.info(f"Symbol processing log saved to {symbols_log_path}")

            # Generate XML tree
            tree = ET.ElementTree(score_partwise)

            # Determine output path
            if output_path is None:
                base_filename = f"{work_title.replace(' ', '_')}.mxl"
                output_path = os.path.join(self.output_dir, base_filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Intermediate XML debug
            debug_xml_path = os.path.join(DEBUG_OUTPUT_DIR, "debug_musicxml.xml")
            tree.write(debug_xml_path, encoding="UTF-8", xml_declaration=True)
            self.logger.info(f"Debug XML saved to {debug_xml_path}")

            # Write final XML file
            tree.write(output_path, encoding="UTF-8", xml_declaration=True)

            self.logger.info(f"MusicXML exported: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"MusicXML export failed: {e}")
            raise

    def _create_note_element(
        self,
        symbol_info: Dict[str, Any],
        staff_context: Optional[Dict[str, Any]] = None,
        advanced_processing: Optional[Dict[str, Any]] = None,
    ) -> ET.Element:
        """
        Create a comprehensive MusicXML note element with advanced processing

        Args:
            symbol_info (Dict[str, Any]): Detailed symbol recognition information
            staff_context (Dict[str, Any], optional): Staff-level contextual information
            advanced_processing (Dict[str, Any], optional): Additional processing configuration

        Returns:
            ET.Element: Enhanced MusicXML note element with metadata and annotations

        Features:
            - Confidence-based annotations
            - Raw symbol image metadata
            - Adaptive pitch selection
            - Advanced processing metadata
        """
        note = ET.Element("note")

        # Normalize symbol information
        symbol_label = (
            symbol_info.get("label", "quarter_note")
            if isinstance(symbol_info, dict)
            else symbol_info
        )
        confidence = (
            symbol_info.get("confidence", 1.0) if isinstance(symbol_info, dict) else 1.0
        )
        raw_image = (
            symbol_info.get("raw_image", None)
            if isinstance(symbol_info, dict)
            else None
        )

        # Pitch mapping
        pitch_map = {
            "C4": {"step": "C", "octave": "4"},
            "D4": {"step": "D", "octave": "4"},
            "E4": {"step": "E", "octave": "4"},
            "F4": {"step": "F", "octave": "4"},
            "G4": {"step": "G", "octave": "4"},
            "A4": {"step": "A", "octave": "4"},
            "B4": {"step": "B", "octave": "4"},
        }

        # Map symbol labels to MusicXML representation
        symbol_map = {
            "quarter_note": ("quarter", 1, False),
            "half_note": ("half", 2, False),
            "whole_note": ("whole", 4, False),
            "eighth_note": ("eighth", 0.5, False),
            "quarter_rest": ("quarter", 1, True),
            "half_rest": ("half", 2, True),
        }

        # Default to quarter note if not recognized
        type_info = symbol_map.get(symbol_label, symbol_map["quarter_note"])

        # Pitch selection
        default_pitch = "C4"

        # Pitch
        if not type_info[2]:  # Not a rest
            pitch_elem = ET.SubElement(note, "pitch")
            pitch_details = pitch_map.get(default_pitch, pitch_map["C4"])
            ET.SubElement(pitch_elem, "step").text = pitch_details["step"]
            ET.SubElement(pitch_elem, "octave").text = pitch_details["octave"]

        # Note type
        ET.SubElement(note, "type").text = type_info[0]

        # Duration
        ET.SubElement(note, "duration").text = str(type_info[1])

        # Rest handling
        if type_info[2]:
            ET.SubElement(note, "rest")

        # Additional notation
        if type_info[0] in ["quarter", "half", "whole"]:
            stem = ET.SubElement(note, "stem")
            stem.text = "up"

        # Confidence annotation
        if confidence < 0.7:  # Low confidence threshold
            annotation = ET.SubElement(note, "annotation")
            annotation.text = f"Low Confidence: {confidence:.2f}"

        # Raw image metadata
        if raw_image is not None:
            # Save symbol image
            image_path = self._save_symbol_image(raw_image, symbol_label)
            metadata = ET.SubElement(note, "metadata")
            ET.SubElement(metadata, "image-path").text = image_path

        # Advanced processing features
        if advanced_processing:
            # Add advanced processing metadata
            for key, value in advanced_processing.items():
                adv_elem = ET.SubElement(note, f"advanced_{key}")
                adv_elem.text = str(value)

        return note

    def _save_symbol_image(self, symbol_image: np.ndarray, symbol_label: str) -> str:
        """
        Save symbol image for reference

        Args:
            symbol_image (np.ndarray): Symbol image
            symbol_label (str): Symbol label

        Returns:
            str: Path to saved image
        """
        # Create symbol image directory
        symbol_dir = os.path.join(DEBUG_OUTPUT_DIR, "symbol_images")
        os.makedirs(symbol_dir, exist_ok=True)

        # Generate unique filename
        filename = f"{symbol_label}_{int(time.time() * 1000)}.png"
        filepath = os.path.join(symbol_dir, filename)

        # Save image
        cv2.imwrite(filepath, symbol_image)

        return filepath


def main():
    # Example usage
    sample_omr_results = {
        "pdf_path": "/path/to/sample.pdf",
        "page_results": [
            {"symbols": [{"label": "quarter_note"}, {"label": "half_note"}]}
        ],
    }

    converter = MusicXMLConverter()
    converter.convert(sample_omr_results)


if __name__ == "__main__":
    main()
