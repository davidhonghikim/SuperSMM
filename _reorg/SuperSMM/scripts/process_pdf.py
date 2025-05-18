#!/usr/bin/env python3
"""Script to process PDF sheet music with the OMR pipeline.

This script processes a PDF file through the complete OMR pipeline, including:
- PDF to image conversion
- Image preprocessing
- Staff line removal
- Symbol segmentation
- Symbol recognition
- HMM decoding
- Output generation (JSON, CSV, MusicXML, MIDI)
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

# Import modular components
from core.output import DirectoryManager
from core.processing import PipelineManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process PDF sheet music with the OMR pipeline.'
    )
    parser.add_argument(
        '--pdf', '-p',
        default='data/input/scores/Somewhere_Over_the_Rainbow.pdf',
        help='Path to the PDF file to process'
    )
    parser.add_argument(
        '--model', '-m',
        default='ml/models/symbol_recognition.h5',
        help='Path to the symbol recognition model'
    )
    parser.add_argument(
        '--vocab', '-v',
        default='ml/models/vocabulary_semantic.txt',
        help='Path to the vocabulary file'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / args.pdf
    model_path = project_root / args.model
    vocab_path = project_root / args.vocab
    
    # Validate paths
    if not pdf_path.exists():
        logging.error(f"PDF file not found: {pdf_path}")
        return 1
    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        return 1
    if not vocab_path.exists():
        logging.error(f"Vocabulary file not found: {vocab_path}")
        return 1
    
    # Create output directory structure
    dir_manager = DirectoryManager()
    output_root = dir_manager.create_output_structure(project_root, pdf_path.stem)
    
    # Initialize pipeline manager
    pipeline_manager = PipelineManager(
        model_path=str(model_path),
        vocab_path=str(vocab_path)
    )
    
    # Process PDF
    result = pipeline_manager.process_pdf(str(pdf_path), output_root)
    
    # Print summary
    print("\nProcessing Results:")
    print("==================")
    
    # Print page statistics
    for page in result['results']:
        if 'error' in page:
            print(f"Error: {page['error']}")
            continue
        print(f"\nPage {page['page_number']}:")
        print(f"  Size: {page['page_size']}")
        print(f"  Staff Lines: {len(page['staff_lines'])}")
        print(f"  Symbols: {len(page['symbols'])}")
        
        # Print first few symbols
        if page['symbols']:
            print("  First few symbols:")
            for symbol in page['symbols'][:3]:
                print(f"    - Position: {symbol['position']}")
                if 'label' in symbol:
                    print(f"      Label: {symbol['label']} (confidence: {symbol['confidence']:.2f})")
    
    # Print output paths
    print("\nOutput Files:")
    print(f"  - Summary JSON: {result['output_paths']['summary_json']}")
    if 'symbols_csv' in result['output_paths']:
        print(f"  - Symbols CSV: {result['output_paths']['symbols_csv']}")
    if 'musicxml' in result['output_paths']:
        print(f"  - MusicXML: {result['output_paths']['musicxml']}")
    if 'midi' in result['output_paths']:
        print(f"  - MIDI: {result['output_paths']['midi']}")
    print(f"  - Images: {output_root / 'processed'}")
    
    print("\nProcessing complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
