"""Test script for the OMR Pipeline."""
import os
import logging
from pathlib import Path
from supersmm.core.pipeline.omr_pipeline import OMRPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_omr_pipeline():
    """Test the OMR pipeline with a sample PDF file."""
    try:
        # Initialize the pipeline
        logger.info("Initializing OMR pipeline...")
        pipeline = OMRPipeline()
        
        # Path to test PDF
        test_pdf = Path("tests/test_data/sample_sheet_music.pdf")
        if not test_pdf.exists():
            raise FileNotFoundError(f"Test file not found: {test_pdf}")
            
        logger.info(f"Processing test file: {test_pdf}")
        
        # Process the PDF
        results = pipeline.process_sheet_music(
            input_path_or_image=str(test_pdf),
            output_dir_base="test_output"
        )
        
        # Print summary
        logger.info("Processing completed successfully!")
        logger.info(f"Pages processed: {len(results.get('pages', []))}")
        
        # Print performance metrics
        metrics = pipeline.get_performance_report()
        logger.info("\nPerformance Metrics:")
        for key, value in metrics.items():
            if key != 'timestamp':
                logger.info(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_omr_pipeline()
    exit(0 if success else 1)
