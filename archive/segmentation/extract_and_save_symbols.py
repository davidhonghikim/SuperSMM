import os
from pathlib import Path
import csv
import cv2
import numpy as np
from symbol_segmenter import SymbolSegmenter


def extract_and_save_symbols_for_page(
    page_image_path, output_dir, segmenter_config=None
):
    """
    Extracts symbols from a processed binary page image and saves them as PNGs in output_dir.
    Args:
        page_image_path (str or Path): Path to the binary processed page image.
        output_dir (str or Path): Directory to save extracted symbol images.
        segmenter_config (dict, optional): Configuration for SymbolSegmenter.
    """
    page_image_path = Path(page_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(page_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[ERROR] Could not load image: {page_image_path}")
        return []

    # Morphological preprocessing to break up connected symbols
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Segment symbols using the SymbolSegmenter's refined logic
    segmenter = SymbolSegmenter(segmenter_config)
    # Ensure segment_symbols is called, which internally uses find_connected_components and then filters
    seg_out = segmenter.segment_symbols(morph_img)
    symbol_crops, text_crops_with_bboxes = (
        seg_out  # Expecting (list_of_symbol_ndarrays, list_of_text_tuples_with_bboxes)
    )

    # Debug: Save image with bounding boxes for detected symbols and text
    debug_img = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)

    # Save symbol candidates
    saved_symbol_paths = []
    for idx, crop_img in enumerate(symbol_crops):
        # To draw bounding box for symbols, we need their original coordinates.
        # This requires segment_symbols to also return bboxes or for us to re-find them.
        # For simplicity in this revert, we'll focus on saving. Bbox drawing for symbols might be missing or inaccurate if not returned by segment_symbols.

        # Resize to height 128px, pad width if needed (standard preprocessing for some models)
        h_orig, w_orig = crop_img.shape[:2]
        scale = 128 / h_orig if h_orig > 0 else 0
        new_w = int(w_orig * scale) if scale > 0 else 0
        if new_w > 0 and 128 > 0:
            resized = cv2.resize(crop_img, (new_w, 128), interpolation=cv2.INTER_AREA)
        else:
            resized = crop_img  # Cannot resize if original or target dims are zero

        # Normalize to [0, 255] if not already (common for image saving)
        if resized.max() <= 1.0 and resized.dtype != np.uint8:
            resized = (resized * 255).astype(np.uint8)
        elif resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)

        symbol_path = output_dir / f"symbol_{idx:03d}.png"
        cv2.imwrite(str(symbol_path), resized)
        saved_symbol_paths.append(str(symbol_path))

    # Save text candidates for OCR
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    saved_text_paths = []
    for idx, (text_img, (x, y, w, h)) in enumerate(text_crops_with_bboxes):
        text_path = text_dir / f"text_{idx:03d}.png"
        # Save raw text crop (no resize needed for most OCR)
        if text_img.dtype != np.uint8:
            text_img_to_save = (
                text_img * 255 if text_img.max() <= 1.0 else text_img
            ).astype(np.uint8)
        else:
            text_img_to_save = text_img
        cv2.imwrite(str(text_path), text_img_to_save)
        saved_text_paths.append(str(text_path))
        # Draw bbox for text
        cv2.rectangle(
            debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1
        )  # Green for text

    # Note: Bounding boxes for symbols are not drawn here as segment_symbols might only return crops.
    # If bboxes for symbols are needed on debug_img, segment_symbols must return them or contours must be re-processed here.
    debug_path = output_dir / "debug_bboxes.png"
    cv2.imwrite(str(debug_path), debug_img)

    print(f"[INFO] Saved {len(saved_symbol_paths)} symbols to {output_dir}")
    print(f"[INFO] Saved {len(saved_text_paths)} text regions to {text_dir}")
    print(
        f"[DEBUG] Saved bounding box debug image to {debug_path} (text boxes in green)"
    )

    # Return paths of saved *symbols* for potential downstream processing
    return saved_symbol_paths


def extract_all_pages(input_dir, output_base_dir):
    """
    Extract symbols from all binary images in input_dir, saving to output_base_dir/page_X/
    """
    input_dir = Path(input_dir)
    output_base_dir = Path(output_base_dir)
    for page_dir in sorted(input_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        staff_removed_path = page_dir / "staff_removed.png"
        if not staff_removed_path.exists():
            print(f"[WARN] No staff_removed.png in {page_dir}")
            continue
        output_dir = output_base_dir / page_dir.name
        segmenter_config = {
            "min_symbol_size": 5,
            "max_symbol_size": 5000,
            "segmentation_strategy": "connected_components",
        }
        extract_and_save_symbols_for_page(
            staff_removed_path, output_dir, segmenter_config=segmenter_config
        )


if __name__ == "__main__":
    # Example usage:
    # Extract all symbols from processed pages and save to /data/output/symbols/
    # Get project root (SuperSMM) by walking up from this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # src/segmentation/ -> src/ -> SuperSMM
    input_dir = project_root / "data/output/symbol_recognition/processed"
    output_base_dir = project_root / "data/output/symbols"
    extract_all_pages(input_dir, output_base_dir)
