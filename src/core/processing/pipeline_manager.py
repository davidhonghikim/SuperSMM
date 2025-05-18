"""
Pipeline Manager Module

Coordinates the complete OMR processing pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.processors.omr_processor import LocalOMRProcessor
from recognition.pipeline import RecognitionPipeline
from core.export.music_exporter import export_musicxml, export_midi
from core.export.musicxml_utils import symbols_to_score
from core.output.output_manager import OutputManager


class PipelineManager:
    """Manages the complete OMR processing pipeline."""

    def __init__(
        self, model_path: Optional[str] = None, vocab_path: Optional[str] = None
    ):
        """Initialize the pipeline manager.

        Args:
            model_path: Optional path to the symbol recognition model
            vocab_path: Optional path to the vocabulary file
        """
        self.logger = logging.getLogger(__name__)

        # Initialize pipeline components
        self.processor = LocalOMRProcessor()
        self.recognition_pipeline = RecognitionPipeline(
            model_path=model_path, vocab_path=vocab_path
        )

    def process_pdf(self, pdf_path: str, output_root: Path) -> Dict[str, Any]:
        """Process a PDF through the complete OMR pipeline.

        Args:
            pdf_path: Path to the PDF file
            output_root: Root directory for outputs

        Returns:
            Dictionary with processing results and output paths
        """
        pdf_name = Path(pdf_path).stem
        output_manager = OutputManager(output_root, pdf_name)

        # Step 1: Process PDF through OMR pipeline
        self.logger.info(f"Processing PDF: {pdf_path}")
        results = self.processor.process_sheet_music(pdf_path)

        # Step 2: Apply symbol recognition and HMM decoding
        self.logger.info("Applying symbol recognition and HMM decoding")
        processed_results = self.recognition_pipeline.process_sheet_music(results)

        # Step 3: Save all outputs
        self.logger.info("Saving outputs")
        output_paths = {"pages": [], "pdf_name": pdf_name}

        # Save page images and metadata
        for result in processed_results:
            if "error" in result:
                self.logger.error(f"Error processing page: {result['error']}")
                continue

            page_num = result["page_number"]
            page_dir = output_manager.save_page_images(page_num, result)
            output_paths["pages"].append(str(page_dir))

        # Save summary JSON
        json_path = output_manager.save_summary_json(processed_results)
        output_paths["summary_json"] = str(json_path)

        # Save symbols CSV
        csv_path = output_manager.save_symbols_csv(processed_results)
        if csv_path:
            output_paths["symbols_csv"] = str(csv_path)

        # Export MusicXML and MIDI
        symbols_per_page = []
        for result in processed_results:
            if "error" not in result and "symbols" in result:
                symbols_per_page.append(result["symbols"])

        if symbols_per_page:
            # Export MusicXML
            musicxml_path = output_root / f"{pdf_name}.musicxml.xml"
            # Flatten symbols_per_page (list of lists) to a flat list of symbol dicts
            flat_symbols = [s for page in symbols_per_page for s in page]
            # Convert recognized symbols to a music21 Score before exporting
            score = symbols_to_score(flat_symbols, title=pdf_name)
            self.logger.info(f"Exporting MusicXML to {musicxml_path}")
            export_musicxml(score, musicxml_path, title=pdf_name)
            output_paths["musicxml"] = str(musicxml_path)

            # Export MIDI
            midi_path = output_root / f"{pdf_name}.music.mid"
            self.logger.info(f"Exporting MIDI to {midi_path}")
            export_midi(score, midi_path)
            output_paths["midi"] = str(midi_path)

        return {"results": processed_results, "output_paths": output_paths}
