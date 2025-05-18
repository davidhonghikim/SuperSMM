#!/usr/bin/env python3
"""
OMR Pipeline Test Script

This script demonstrates how to use the OMR pipeline to process sheet music.
"""

import os
import sys
import logging
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.supersmm.core.pipeline import create_omr_pipeline
except ImportError as e:
    print(f"Error importing OMR pipeline: {e}")
    print("Trying to install required packages...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    from src.supersmm.core.pipeline import create_omr_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
    page_numbers: Optional[List[int]] = None,
) -> List[str]:
    """Convert PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output images
        dpi: DPI for the output images
        page_numbers: List of page numbers to convert (1-based), or None for all pages
        
    Returns:
        List of paths to the generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    
    # If no specific pages are specified, process all pages
    if page_numbers is None:
        page_numbers = list(range(1, len(doc) + 1))
    
    for page_num in page_numbers:
        # Convert to 0-based index
        idx = page_num - 1
        if idx < 0 or idx >= len(doc):
            logger.warning(f"Page {page_num} is out of range. Skipping...")
            continue
            
        # Get the page
        page = doc[idx]
        
        # Convert to image (RGB)
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            (pix.height, pix.width, pix.n)
        )
        
        # Convert from RGB to BGR for OpenCV
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save the image
        output_path = os.path.join(
            output_dir,
            f"{Path(pdf_path).stem}_page{page_num:03d}.png"
        )
        cv2.imwrite(output_path, img)
        image_paths.append(output_path)
        logger.info(f"Saved page {page_num} to {output_path}")
    
    doc.close()
    return image_paths


def process_sheet_music(
    input_path: str,
    output_dir: str,
    dpi: int = 300,
    page_numbers: Optional[List[int]] = None,
    model_path: Optional[str] = None,
    use_gpu: bool = False,
    debug: bool = True,
) -> None:
    """Process sheet music file with the OMR pipeline.
    
    Args:
        input_path: Path to the input file (PDF or image)
        output_dir: Directory to save the output files
        dpi: DPI for PDF to image conversion
        page_numbers: List of page numbers to process (1-based), or None for all pages
        model_path: Path to the symbol recognition model
        use_gpu: Whether to use GPU for inference
        debug: Whether to enable debug mode
    """
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert PDF to images if needed
    input_path = Path(input_path)
    if input_path.suffix.lower() == '.pdf':
        logger.info(f"Converting PDF to images: {input_path}")
        image_paths = convert_pdf_to_images(
            str(input_path),
            images_dir,
            dpi=dpi,
            page_numbers=page_numbers,
        )
    else:
        # For single image files
        image_paths = [str(input_path)]
    
    if not image_paths:
        logger.error("No images to process.")
        return
    
    # Initialize the OMR pipeline
    logger.info("Initializing OMR pipeline...")
    pipeline = create_omr_pipeline(
        model_path=model_path,
        use_gpu=use_gpu,
        debug=debug,
        output_dir=results_dir,
    )
    
    # Process each image
    logger.info(f"Processing {len(image_paths)} images...")
    results = pipeline.process_batch(image_paths)
    
    # Log summary
    logger.info("\n=== Processing Complete ===")
    logger.info(f"Processed {len(results)} images")
    logger.info(f"Results saved to: {results_dir}")
    
    # Print statistics
    stats = pipeline.get_statistics()
    logger.info("\n=== Statistics ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process sheet music with OMR.")
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input file (PDF or image)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory (default: 'output')",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page numbers to process (e.g., '1,3,5' or '1-5' or '1,3-5,7')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the symbol recognition model",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Parse page numbers
    page_numbers = None
    if args.pages:
        page_numbers = []
        parts = args.pages.split(',')
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_numbers.extend(range(start, end + 1))
            else:
                page_numbers.append(int(part))
        # Remove duplicates and sort
        page_numbers = sorted(list(set(page_numbers)))
    
    # Process the sheet music
    process_sheet_music(
        input_path=args.input,
        output_dir=args.output,
        dpi=args.dpi,
        page_numbers=page_numbers,
        model_path=args.model,
        use_gpu=args.gpu,
        debug=args.debug,
    )
