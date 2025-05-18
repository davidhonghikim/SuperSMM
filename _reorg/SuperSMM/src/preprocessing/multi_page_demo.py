import sys
import os
import logging
from advanced_preprocessor import AdvancedPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_multi_page_processing():
    """
    Demonstrate multi-page PDF processing capabilities
    """
    preprocessor = AdvancedPreprocessor()

    # Sample PDF for testing
    sample_pdf = "/Users/danger/CascadeProjects/LOO/SheetMasterMusic/sample_files/pdfs/Somewhere Over the Rainbow.pdf"

    # Test scenarios
    scenarios = [
        {"description": "Process First Page Only", "page_range": 0},
        {"description": "Process First Three Pages", "page_range": (0, 2)},
        {"description": "Process All Pages", "page_range": None},
    ]

    for scenario in scenarios:
        try:
            logger.info(f"Processing Scenario: {scenario['description']}")

            # Process PDF pages
            processed_pages = preprocessor.process_pdf(
                sample_pdf, page_range=scenario["page_range"]
            )

            # Log processing results
            logger.info(f"Processed {len(processed_pages)} pages")

            for page_data in processed_pages:
                page_num = page_data.get("page_number", "Unknown")
                logger.info(f"Page {page_num}:")

                # Optional: Save processed images
                output_dir = "/tmp/processed_pages"
                os.makedirs(output_dir, exist_ok=True)

                for key, image in page_data.items():
                    if key != "page_number":
                        try:
                            logger.info(f"  {key} image shape: {image.shape}")

                            output_path = os.path.join(
                                output_dir, f"page_{page_num}_{key}.png"
                            )
                            import cv2

                            cv2.imwrite(output_path, image)
                            logger.info(f"  Saved {key} to {output_path}")
                        except Exception as img_err:
                            logger.error(
                                f"Error processing {key} for page {page_num}: {img_err}"
                            )

        except Exception as e:
            logger.error(f"Error in scenario {scenario['description']}: {e}")


if __name__ == "__main__":
    demo_multi_page_processing()
