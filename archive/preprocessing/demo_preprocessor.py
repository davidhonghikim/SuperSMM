import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.advanced_preprocessor import AdvancedPreprocessor
import numpy as np
import cv2

# Use the centralized logger
from src.utils.logger import setup_logger
logger = setup_logger("preprocessing.demo")


def demo_preprocessing():
    """
    Demonstrate the advanced preprocessing capabilities
    with different input types and error handling
    """
    preprocessor = AdvancedPreprocessor()

    # Test cases
    # Verify sample files exist
    sample_pdf = "/Users/danger/CascadeProjects/LOO/SheetMasterMusic/sample_files/pdfs/Somewhere Over the Rainbow.pdf"
    sample_image = "/path/to/sample_image.png"  # Replace with actual path

    test_cases = [
        # Test case 1: NumPy array input
        {
            "input": np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),
            "description": "NumPy Array Input",
        },
        # Test case 2: PDF file path
        {
            "input": sample_pdf if os.path.exists(sample_pdf) else None,
            "description": "PDF File Input",
        },
        # Test case 3: PNG/JPG image file
        {
            "input": sample_image if os.path.exists(sample_image) else None,
            "description": "Image File Input",
        },
    ]

    # Remove test cases with non-existent files
    test_cases = [case for case in test_cases if case["input"] is not None]

    for case in test_cases:
        try:
            logger.info(f"Processing: {case['description']}")

            # Process the page
            result = preprocessor.process_page(case["input"])

            # Visualize results (optional)
            for key, image in result.items():
                if isinstance(image, np.ndarray):
                    logger.info(f"{key} image shape: {image.shape}")

                    # Optional: Save processed images for inspection
                    output_path = f"/tmp/processed_{key}.png"
                    cv2.imwrite(output_path, image)
                    logger.info(f"Saved {key} image to {output_path}")

        except Exception as e:
            logger.error(f"Error processing {case['description']}: {e}")


if __name__ == "__main__":
    demo_preprocessing()
