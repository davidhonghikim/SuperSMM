import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import fitz  # PyMuPDF

# Initialize a module-level logger
logger = logging.getLogger(__name__)


def _determine_page_indices(
    num_pages_in_doc: int,
    page_range_config: Union[int, Tuple[int, int], None],
    pdf_name_for_logging: str,
    log: logging.Logger,
) -> Tuple[int, int]:
    """
    Determines the start and end page indices for processing based on the configuration.

    Args:
        num_pages_in_doc (int): Total number of pages in the PDF.
        page_range_config (Union[int, Tuple[int, int], None]): User-defined page range.
        pdf_name_for_logging (str): Name of the PDF file for logging.
        log (logging.Logger): Logger instance.

    Returns:
        Tuple[int, int]: (start_page_idx, end_page_idx).

    Raises:
        ValueError: If page_range_config is invalid.
        TypeError: If page_range_config has an invalid type.
    """
    start_page_idx, end_page_idx = 0, num_pages_in_doc - 1  # Default to all pages

    if isinstance(page_range_config, int):
        if not (0 <= page_range_config < num_pages_in_doc):
            err_msg = f"Invalid single page number for {pdf_name_for_logging}: {page_range_config}. PDF has {num_pages_in_doc} pages (0-indexed)."
            log.error(err_msg)
            raise ValueError(err_msg)
        start_page_idx = end_page_idx = page_range_config
    elif isinstance(page_range_config, tuple) and len(page_range_config) == 2:
        s, e = page_range_config
        if not (0 <= s < num_pages_in_doc and 0 <= e < num_pages_in_doc and s <= e):
            err_msg = f"Invalid page range tuple for {pdf_name_for_logging}: {(s, e)}. PDF has {num_pages_in_doc} pages (0-indexed)."
            log.error(err_msg)
            raise ValueError(err_msg)
        start_page_idx, end_page_idx = s, e
    elif page_range_config is not None:  # Invalid format for page_range_config
        err_msg = f"page_range_config for {pdf_name_for_logging} must be int, tuple(int, int), or None."
        log.error(err_msg)
        raise TypeError(err_msg)
    return start_page_idx, end_page_idx


def _convert_page_to_bgr_image(
    page: fitz.Page,
    page_idx_0_based: int,
    pdf_name_for_logging: str,
    zoom_factor: float,
    log: logging.Logger,
) -> Optional[np.ndarray]:
    """
    Converts a fitz.Page object to a BGR NumPy array.

    Args:
        page (fitz.Page): The PDF page object from PyMuPDF.
        page_idx_0_based (int): The 0-indexed page number (for logging).
        pdf_name_for_logging (str): Name of the PDF file (for logging).
        zoom_factor (float): Zoom factor to use for rendering the page to a pixmap.
        log (logging.Logger): Logger instance.

    Returns:
        Optional[np.ndarray]: The BGR image as a NumPy array, or None if conversion fails.
    """
    try:
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False for RGB

        if pix.width == 0 or pix.height == 0:
            log.error(
                f"Page {page_idx_0_based} from PDF '{pdf_name_for_logging}' resulted in an empty pixmap."
            )
            return None

        image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )

        if pix.n == 3:  # RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:  # Grayscale
            image_np = cv2.cvtColor(
                image_np, cv2.COLOR_GRAY2BGR
            )  # Convert to BGR for consistency
        elif pix.n == 4:  # RGBA (or CMYK converted by fitz to RGBA)
            log.warning(
                f"Page {page_idx_0_based} of '{pdf_name_for_logging}' has {pix.n} components (e.g., RGBA). Attempting conversion to BGR."
            )
            try:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            except cv2.error as cv_err:
                log.error(
                    f"Could not convert page {page_idx_0_based} (shape {image_np.shape}) of '{pdf_name_for_logging}' from RGBA to BGR: {cv_err}. Skipping image conversion."
                )
                return None
        else:
            log.error(
                f"Page {page_idx_0_based} of '{pdf_name_for_logging}' has unexpected number of components: {pix.n}. Skipping image conversion."
            )
            return None

        log.debug(
            f"Page {page_idx_0_based} of '{pdf_name_for_logging}' converted to image, shape: {image_np.shape}"
        )
        return image_np

    except Exception as e:
        log.error(
            f"Error converting page {page_idx_0_based} of '{pdf_name_for_logging}' to image: {e}",
            exc_info=True,
        )
        return None


def extract_and_process_pdf_pages(
    pdf_path: str,
    process_page_func: Callable[..., Dict[str, Any]],
    logger_instance: logging.Logger,
    output_dir_base: Optional[str] = None,
    page_range: Union[int, Tuple[int, int], None] = None,
    zoom_factor: float = 2.0,  # DPI scaling factor, 2.0 ~ 192 DPI
) -> List[Dict[str, Any]]:
    """
    Extracts pages from a PDF, converts them to images, and processes them using a provided function.

    Args:
        pdf_path (str): Path to the PDF file.
        process_page_func (Callable): The function to call for processing each extracted page image.
                                     Expected signature: process_page(image_input: np.ndarray,
                                                                       output_dir: Optional[str],
                                                                       page_identifier: str) -> Dict[str, Any]
        logger_instance (logging.Logger): Logger for logging messages.
        output_dir_base (Optional[str]): Base directory to save processed images.
                                         A subdirectory named after the PDF will be created here.
        page_range (Union[int, Tuple[int, int], None]):
            - If int: Process only that specific page (0-indexed).
            - If tuple (start, end): Process pages in that range (inclusive).
            - If None: Process all pages.
        zoom_factor (float): Zoom factor for rendering PDF pages to images.
                             Higher values increase DPI. Default is 2.0 (approx 192 DPI).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                              the processing results for a page.
    """
    log = logger_instance or logger
    pdf_name_for_logging = Path(pdf_path).name  # Use for logging consistency

    if not os.path.exists(pdf_path):
        log.error(f"PDF path does not exist: {pdf_path}")
        raise FileNotFoundError(f"PDF path does not exist: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error(f"Could not open PDF {pdf_path}: {e}", exc_info=True)
        raise ValueError(
            f"Could not open PDF {pdf_path}: {e}"
        )  # Re-raise to signal failure

    num_pages_in_doc = len(doc)
    if num_pages_in_doc == 0:
        log.warning(f"PDF '{pdf_name_for_logging}' contains no pages.")
        doc.close()
        return []

    try:
        start_page_idx, end_page_idx = _determine_page_indices(
            num_pages_in_doc, page_range, pdf_name_for_logging, log
        )
    except (ValueError, TypeError) as e:  # Errors already logged by helper
        doc.close()
        # Re-raise to signal failure to the caller, allowing them to handle it appropriately.
        # For example, if this is called in a batch script, it might skip this PDF.
        raise

    log.info(
        f"Processing PDF '{pdf_name_for_logging}', pages {start_page_idx} to {end_page_idx} (0-indexed). Total pages in doc: {num_pages_in_doc}."
    )

    all_pages_results: List[Dict[str, Any]] = []
    pdf_specific_output_dir_path: Optional[Path] = None

    if output_dir_base:
        pdf_stem = Path(pdf_path).stem
        pdf_specific_output_dir_path = Path(output_dir_base) / pdf_stem
        try:
            pdf_specific_output_dir_path.mkdir(parents=True, exist_ok=True)
            log.info(
                f"Output for PDF '{pdf_stem}' will be saved in: {pdf_specific_output_dir_path}"
            )
        except OSError as e:
            log.error(
                f"Could not create output directory {pdf_specific_output_dir_path}: {e}. Images will not be saved for this PDF."
            )
            pdf_specific_output_dir_path = None  # Disable saving

    for i in range(start_page_idx, end_page_idx + 1):
        page_num_display = i + 1  # For user-friendly logging/naming (1-indexed)
        page_identifier = f"page_{i:03d}"
        log.info(
            f"Starting processing for {page_identifier} (Display Page {page_num_display}/{num_pages_in_doc}) from '{pdf_name_for_logging}'..."
        )

        current_page_results: Dict[str, Any] = {
            "page_number_0_indexed": i,
            "page_number_1_indexed": page_num_display,
            "pdf_name": pdf_name_for_logging,  # Store PDF name in results
            "page_identifier": page_identifier,
            "error": None,
            "error_stage": None,
        }

        try:
            page_obj = doc.load_page(i)  # Use fitz.Page object
            image_np = _convert_page_to_bgr_image(
                page_obj, i, pdf_name_for_logging, zoom_factor, log
            )

            if image_np is None:
                # Error already logged by _convert_page_to_bgr_image
                current_page_results.update(
                    {
                        "error": "Page to image conversion failed",
                        "error_stage": "pdf_extraction",
                    }
                )
            else:
                # Call the main processing function (e.g., AdvancedPreprocessor.process_page)
                processed_data = process_page_func(
                    image_input=image_np,
                    output_dir=(
                        str(pdf_specific_output_dir_path)
                        if pdf_specific_output_dir_path
                        else None
                    ),
                    page_identifier=page_identifier,
                )
                current_page_results.update(
                    processed_data
                )  # Merge results from process_page_func

        except Exception as e:
            # This catches errors in the loop itself or from process_page_func if they weren't caught internally
            log.error(
                f"Critical error during processing loop for {page_identifier} of '{pdf_name_for_logging}': {e}",
                exc_info=True,
            )
            current_page_results.update(
                {"error": str(e), "error_stage": "pdf_page_processing_loop"}
            )

        all_pages_results.append(current_page_results)
        log.info(
            f"Finished processing for {page_identifier} of '{pdf_name_for_logging}'. Error: {current_page_results.get('error')}"
        )

    doc.close()
    log.info(
        f"Finished all page processing for PDF '{pdf_name_for_logging}'. {len(all_pages_results)} pages attempted."
    )
    return all_pages_results
