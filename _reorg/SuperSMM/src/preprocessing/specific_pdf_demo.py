import sys
import os
import logging
import numpy as np
import cv2
from advanced_preprocessor import AdvancedPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_specific_pdf_processing():
    """
    Demonstrate PDF processing with a specific file
    """
    preprocessor = AdvancedPreprocessor()

    # Specific PDF file path
    pdf_path = "/Users/danger/CascadeProjects/LOO/SuperSMM/imports/Somewhere_Over_the_Rainbow.pdf"

    # Processing scenarios
    scenarios = [
        {"description": "Process First Page", "page_range": 0},
        {"description": "Process All Pages", "page_range": None},
    ]

    for scenario in scenarios:
        try:
            logger.info(f"Processing Scenario: {scenario['description']}")

            # Process PDF pages
            processed_pages = preprocessor.process_pdf(
                pdf_path, page_range=scenario["page_range"]
            )

            # Log processing results
            logger.info(f"Processed {len(processed_pages)} pages")

            # Create output directory
            output_base_dir = "/tmp/specific_pdf_processing"
            os.makedirs(output_base_dir, exist_ok=True)

            for page_data in processed_pages:
                page_num = page_data.get("page_number", "unknown")
                page_dir = os.path.join(output_base_dir, f"page_{page_num}")
                os.makedirs(page_dir, exist_ok=True)

                logger.info(f"Processing Page {page_num}:")

                for key, image in page_data.items():
                    if key != "page_number":
                        try:
                            # Log image details
                            logger.info(f"  {key} image shape: {image.shape}")

                            # Special handling for staff_lines
                            if key == "staff_lines":
                                # Create a visualization of staff lines
                                staff_line_vis = np.zeros(
                                    (image.shape[0], image.shape[1], 3), dtype=np.uint8
                                )
                                for start_y, end_y in image:
                                    cv2.line(
                                        staff_line_vis,
                                        (0, start_y),
                                        (staff_line_vis.shape[1], end_y),
                                        (0, 255, 0),
                                        2,
                                    )
                                image = staff_line_vis

                            # Save processed images
                            output_path = os.path.join(page_dir, f"{key}.png")
                            cv2.imwrite(output_path, image)
                            logger.info(f"  Saved {key} to {output_path}")

                        except Exception as img_err:
                            logger.error(
                                f"Error processing {key} for page {page_num}: {img_err}"
                            )

        except Exception as e:
            logger.error(f"Error in scenario {scenario['description']}: {e}")


if __name__ == "__main__":
    demo_specific_pdf_processing()
