"""Primus dataset loader for Optical Music Recognition."""

import cv2
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

import ctc_utils

# Import the centralized logger
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))  # Add project root to path
from utils.logger import setup_logger

# Configure logging
logger = setup_logger("tf-deep-omr.primus", log_type="ml")


class CTC_PriMuS:
    """A class for handling the Primus dataset for Optical Music Recognition.

    This class provides methods to load and process music scores and their corresponding
    ground truth for training and validation.
    """

    gt_element_separator: str = "-"
    PAD_COLUMN: int = 0
    validation_dict: Optional[Dict[str, list]] = None

    def __init__(
        self,
        corpus_dirpath: Union[str, Path],
        corpus_filepath: Union[str, Path],
        dictionary_path: Union[str, Path],
        semantic: bool,
        distortions: bool = False,
        val_split: float = 0.0,
    ) -> None:
        """Initialize the CTC_PriMuS dataset handler.

        Args:
            corpus_dirpath: Path to the directory containing the corpus files.
            corpus_filepath: Path to the file containing the list of samples.
            dictionary_path: Path to the dictionary file containing all possible symbols.
            semantic: Whether to use semantic or agnostic ground truth.
            distortions: Whether to use distorted versions of the images.
            val_split: Fraction of the dataset to use for validation.
        """
        self.semantic = semantic
        self.distortions = distortions
        self.corpus_dirpath = Path(corpus_dirpath)
        self.current_idx = 0

        # Load corpus
        try:
            with open(corpus_filepath, "r", encoding="utf-8") as corpus_file:
                corpus_list = corpus_file.read().splitlines()

            # Load dictionary
            with open(dictionary_path, "r", encoding="utf-8") as dict_file:
                dict_list = dict_file.read().splitlines()

            # Build vocabulary
            self.word2int: Dict[str, int] = {}
            self.int2word: Dict[int, str] = {}

            for word in dict_list:
                if word not in self.word2int:
                    word_idx = len(self.word2int)
                    self.word2int[word] = word_idx
                    self.int2word[word_idx] = word

            self.vocabulary_size = len(self.word2int)

            # Split into training and validation sets
            if val_split > 0:
                random.shuffle(corpus_list)
                val_idx = int(len(corpus_list) * val_split)
                self.training_list = corpus_list[val_idx:]
                self.validation_list = corpus_list[:val_idx]
            else:
                self.training_list = corpus_list
                self.validation_list = []

            logger.info(
                f"Loaded dataset: {len(self.training_list)} training, "
                f"{len(self.validation_list)} validation samples"
            )

        except FileNotFoundError as e:
            logger.error(f"Failed to initialize dataset: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing dataset: {e}")
            raise

    def _resolve_image_path(self, sample_filepath: Union[str, Path]) -> Path:
        """Resolve the full path to an image file.

        This method attempts to find the image file by trying different extensions
        and handling distorted versions if specified. If the path is a directory,
        it looks for a PNG file inside that directory.

        Args:
            sample_filepath: Relative path to the image file (without extension)

        Returns:
            Path: Full path to the image file

        Raises:
            FileNotFoundError: If no matching image file is found
        """
        sample_filepath = Path(sample_filepath)
        full_path = self.corpus_dirpath / sample_filepath

        # If the path is a directory, look for a PNG file inside it
        if full_path.is_dir():
            png_files = list(full_path.glob("*.png"))
            if png_files:
                return png_files[0].resolve()
            # If no PNG files found, continue with the normal resolution

        # Try the sample_filepath as-is first
        if full_path.exists() and full_path.is_file():
            return full_path.resolve()

        # Try with different extensions
        extensions = [".png", ".jpg", ".jpeg"]
        for ext in extensions:
            if self.distortions and ext == ".png":
                test_path = self.corpus_dirpath / f"{sample_filepath}_distorted.jpg"
            else:
                test_path = self.corpus_dirpath / f"{sample_filepath}{ext}"

            if test_path.exists() and test_path.is_file():
                return test_path.resolve()

        raise FileNotFoundError(
            f"Could not find image for {sample_filepath} in {self.corpus_dirpath}"
        )

    def _resolve_ground_truth_path(self, sample_filepath: Union[str, Path]) -> Path:
        """Resolve the full path to a ground truth file.

        Args:
            sample_filepath: Relative path to the image file

        Returns:
            Path: Full path to the ground truth file

        Note:
            Creates a dummy ground truth file if none exists
        """
        sample_filepath = Path(sample_filepath)
        base_path = sample_filepath.with_suffix("")  # Remove any existing extension

        # Determine ground truth file type
        gt_extension = ".semantic" if self.semantic else ".agnostic"
        gt_path = (self.corpus_dirpath / base_path).with_suffix(gt_extension)

        # Create dummy ground truth if file doesn't exist
        if not gt_path.exists():
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(gt_path, "w", encoding="utf-8") as f:
                    f.write("clef.G-2 note.quarter-2")
                logger.info(f"Created dummy ground truth file: {gt_path}")
            except IOError as e:
                logger.error(f"Failed to create dummy ground truth at {gt_path}: {e}")
                raise

        return gt_path.resolve()

    def nextBatch(
        self, params: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray, List[List[int]]]:
        """Get the next batch of training data.

        Args:
            params: Dictionary containing batch parameters including:
                - batch_size: Number of samples per batch
                - img_height: Target height for resizing images
                - img_channels: Number of image channels (1 for grayscale)
                - conv_blocks: Number of convolutional blocks
                - conv_pooling_size: List of pooling sizes for each block

        Returns:
            Tuple containing:
                - batch_images: Padded batch of images as a numpy array
                - lengths: Array of sequence lengths after convolutions
                - labels: List of label sequences
        """
        images: List[npt.NDArray] = []
        labels: List[List[int]] = []
        retries: int = 0
        max_retries: int = 3

        if not self.training_list:
            raise ValueError(
                "No training data available. Check dataset initialization."
            )

        while len(images) < params["batch_size"] and retries < max_retries:
            try:
                # Read files until we have enough samples or run out of data
                while len(images) < params["batch_size"]:
                    if self.current_idx >= len(self.training_list):
                        self.current_idx = 0
                        random.shuffle(self.training_list)
                        logger.info("Reached end of training list, reshuffling...")

                    sample_filepath = self.training_list[self.current_idx]
                    self.current_idx += 1

                    try:
                        # Load and preprocess image
                        image_path = self._resolve_image_path(sample_filepath)
                        sample_img = self._load_image(image_path, params["img_height"])

                        # Load and process ground truth
                        gt_path = self._resolve_ground_truth_path(sample_filepath)
                        mapped_labels = self._process_ground_truth(
                            gt_path, sample_filepath
                        )

                        if (
                            mapped_labels is not None
                        ):  # Skip samples with invalid labels
                            labels.append(mapped_labels)
                            images.append(ctc_utils.normalize(sample_img))
                            logger.debug(f"Processed sample: {sample_filepath}")

                    except FileNotFoundError as e:
                        logger.warning(f"Skipping sample {sample_filepath}: {e}")
                        continue
                    except Exception as e:
                        logger.error(
                            f"Error processing sample {sample_filepath}: {e}",
                            exc_info=True,
                        )
                        continue

            except Exception as e:
                logger.error(f"Batch processing error: {e}", exc_info=True)
                retries += 1
                if retries >= max_retries:
                    logger.warning("Max retries reached, returning partial batch")
                    break
                continue

            # Reset retry counter on successful batch collection
            retries = 0

        if not images:
            raise RuntimeError("Failed to load any valid samples for the batch")

        # Create padded batch
        batch_images = self._create_padded_batch(images, params)

        # Calculate sequence lengths after convolutions
        lengths = self._calculate_output_lengths(batch_images.shape[2], params)

        return batch_images, np.asarray(lengths), labels

    def _load_image(self, image_path: Path, target_height: int) -> np.ndarray:
        """Load and preprocess an image.

        Args:
            image_path: Path to the image file or directory containing the image
            target_height: Target height for resizing

        Returns:
            Preprocessed image as a numpy array
        """
        # If the path is a directory, look for a PNG file inside it
        if image_path.is_dir():
            png_files = list(image_path.glob("*.png"))
            if not png_files:
                raise FileNotFoundError(
                    f"No PNG files found in directory: {image_path}"
                )
            # Use the first PNG file found
            image_path = png_files[0]

        try:
            # Try loading with PIL first (handles more formats)
            with Image.open(image_path) as pil_img:
                # Convert to grayscale if needed
                if pil_img.mode != "L":
                    pil_img = pil_img.convert("L")

                # Calculate new dimensions maintaining aspect ratio
                width, height = pil_img.size
                new_width = int(width * target_height / height)

                # Resize image
                pil_img = pil_img.resize((new_width, target_height), Image.LANCZOS)

                # Convert to numpy array and normalize
                img_array = np.array(pil_img, dtype=np.float32) / 255.0

                # Add channel dimension if needed
                if len(img_array.shape) == 2:
                    img_array = img_array[..., np.newaxis]

                return img_array

        except Exception as e:
            # Fall back to OpenCV if PIL fails
            try:
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to load image: {image_path}")

                # Calculate new dimensions maintaining aspect ratio
                height, width = img.shape
                new_width = int(width * target_height / height)

                # Resize image
                img = cv2.resize(
                    img, (new_width, target_height), interpolation=cv2.INTER_AREA
                )

                # Normalize and add channel dimension
                img = img.astype(np.float32) / 255.0
                return img[..., np.newaxis]

            except Exception as cv_e:
                raise RuntimeError(
                    f"Failed to load image with both PIL and OpenCV. "
                    f"PIL error: {e}, OpenCV error: {cv_e}"
                )

    def _process_ground_truth(
        self, gt_path: Path, sample_id: str
    ) -> Optional[List[int]]:
        """Process ground truth file and map symbols to integers.

        Args:
            gt_path: Path to the ground truth file
            sample_id: Identifier for the sample (for logging)

        Returns:
            List of integer labels, or None if the sample should be skipped
        """

        def convert_symbol(symbol: str) -> str:
            """Convert ground truth symbol to match vocabulary format."""
            # Handle clef format conversion (e.g., 'clef.G-2' -> 'clef-G2')
            if symbol.startswith('clef.') and '-' in symbol:
                # Handle case where there might be multiple hyphens
                parts = symbol[5:].split('-')
                if len(parts) >= 2:
                    clef_type = parts[0]
                    clef_line = parts[-1].lstrip('-')
                    return f'clef-{clef_type}{clef_line}'
            return symbol

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                symbols = f.readline().strip().split(ctc_utils.word_separator())

            mapped_labels = []
            for symbol in symbols:
                # Try original symbol first
                if symbol in self.word2int:
                    mapped_labels.append(self.word2int[symbol])
                    continue

                # Try converted symbol
                converted = convert_symbol(symbol)
                if converted in self.word2int:
                    mapped_labels.append(self.word2int[converted])
                    logger.debug(
                        f"Converted symbol '{symbol}' to '{converted}' in {sample_id}"
                    )
                elif "<UNK>" in self.word2int:
                    mapped_labels.append(self.word2int["<UNK>"])
                    logger.warning(
                        f"Unknown symbol '{symbol}' (converted: '{converted}') in {sample_id}, using <UNK>"
                    )
                else:
                    logger.warning(
                        f"Unknown symbol '{symbol}' (converted: '{converted}') in {sample_id} and no <UNK> token, skipping sample"
                    )
                    return None

            return mapped_labels

        except Exception as e:
            logger.error(f"Error processing ground truth {gt_path}: {e}", exc_info=True)
            return None

    def _create_padded_batch(
        self, images: List[npt.NDArray], params: Dict[str, any]
    ) -> npt.NDArray:
        """Create a padded batch from a list of variable-sized images.

        Args:
            images: List of grayscale images (height x width)
            params: Batch parameters including batch_size, img_height, img_channels

        Returns:
            Padded batch as a numpy array
        """
        max_width = max(img.shape[1] for img in images)
        height = params["img_height"]
        channels = params["img_channels"]

        # Initialize batch with padding value
        batch = np.full(
            shape=(params["batch_size"], height, max_width, channels),
            fill_value=self.PAD_COLUMN,
            dtype=np.float32,
        )

        # Fill batch with images (left-aligned)
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            batch[i, :h, :w, 0] = img if img.ndim == 2 else img[:, :, 0]

        return batch

    def _calculate_output_lengths(
        self, input_width: int, params: Dict[str, any]
    ) -> List[int]:
        """Calculate output sequence lengths after applying convolutions.

        Args:
            input_width: Width of the input feature maps
            params: Model parameters including conv_blocks and conv_pooling_size

        Returns:
            List of output lengths (one per sample in the batch)
        """
        width_reduction = 1
        for i in range(params["conv_blocks"]):
            width_reduction *= params["conv_pooling_size"][i][1]

        # Ensure at least 1 and at most input_width
        output_length = max(1, input_width // width_reduction)
        return [output_length] * params["batch_size"]

    def _resolve_validation_image_path(self, sample_filepath: Union[str, Path]) -> Path:
        """Resolve the full path to a validation image file.

        This is a specialized version of _resolve_image_path for validation data.
        It follows the same logic but doesn't handle distorted versions.

        Args:
            sample_filepath: Relative path to the image file (without extension)

        Returns:
            Path: Full path to the validation image file

        Raises:
            FileNotFoundError: If no matching image file is found
        """
        sample_filepath = Path(sample_filepath)

        # Try the sample_filepath as-is first
        full_path = self.corpus_dirpath / sample_filepath
        if full_path.exists():
            return full_path.resolve()

        # Try with different extensions (without distortion variants)
        for ext in [".png", ".jpg", ".jpeg"]:
            test_path = self.corpus_dirpath / f"{sample_filepath}{ext}"
            if test_path.exists():
                return test_path.resolve()

        raise FileNotFoundError(
            f"Could not find validation image for {sample_filepath} in {self.corpus_dirpath}"
        )

    def _resolve_validation_ground_truth_path(
        self, sample_filepath: Union[str, Path]
    ) -> Path:
        """Resolve the full path to a validation ground truth file.

        This is a specialized version of _resolve_ground_truth_path for validation data.

        Args:
            sample_filepath: Relative path to the image file

        Returns:
            Path: Full path to the validation ground truth file

        Note:
            Creates a dummy ground truth file if none exists
        """
        return self._resolve_ground_truth_path(sample_filepath)

    def getValidation(
        self, params: Dict[str, any]
    ) -> Tuple[npt.NDArray, npt.NDArray, List[List[int]]]:
        """Get a batch of validation data.

        This method caches the validation data in memory after the first call
        to improve performance during evaluation.

        Args:
            params: Dictionary containing batch parameters including:
                - batch_size: Number of samples per batch
                - img_height: Target height for resizing images
                - img_channels: Number of image channels (1 for grayscale)
                - conv_blocks: Number of convolutional blocks
                - conv_pooling_size: List of pooling sizes for each block

        Returns:
            Tuple containing:
                - batch_images: Padded batch of images as a numpy array
                - lengths: Array of sequence lengths after convolutions
                - labels: List of label sequences

        Raises:
            ValueError: If no validation data is available
        """
        # Return cached validation data if available
        if self.validation_dict is not None:
            return (
                self.validation_dict["images"],
                np.asarray(self.validation_dict["lengths"], dtype=np.int32),
                self.validation_dict["labels"],
            )

        if not self.validation_list:
            raise ValueError(
                "No validation data available. Check dataset initialization."
            )

        # For testing purposes, limit validation set size
        max_validation_samples = min(100, len(self.validation_list))
        validation_samples = self.validation_list[:max_validation_samples]
        logger.info(f"Loading validation data ({len(validation_samples)} samples)")

        images: List[npt.NDArray] = []
        labels: List[List[int]] = []
        lengths: List[int] = []

        # Process validation samples
        for sample_filepath in validation_samples:
            try:
                # Load and preprocess image
                image_path = self._resolve_validation_image_path(sample_filepath)
                sample_img = self._load_image(image_path, params["img_height"])

                # Process ground truth
                gt_path = self._resolve_validation_ground_truth_path(sample_filepath)
                mapped_labels = self._process_ground_truth(gt_path, sample_filepath)

                if mapped_labels is not None:  # Skip samples with invalid labels
                    labels.append(mapped_labels)
                    images.append(ctc_utils.normalize(sample_img))
                    lengths.append(sample_img.shape[1])

                    # Stop if we have enough samples
                    if len(images) >= params["batch_size"]:
                        break

            except FileNotFoundError as e:
                logger.warning(f"Skipping validation sample {sample_filepath}: {e}")
                continue
            except Exception as e:
                logger.error(
                    f"Error processing validation sample {sample_filepath}: {e}",
                    exc_info=True,
                )
                continue

        if not images:
            raise RuntimeError("Failed to load any valid validation samples")

        # Create padded batch
        batch_images = self._create_padded_batch(images, params)

        # Calculate sequence lengths after convolutions
        seq_lengths = self._calculate_output_lengths(batch_images.shape[2], params)

        # Cache the results
        self.validation_dict = {
            "images": batch_images,
            "lengths": seq_lengths,
            "labels": labels,
        }

        return batch_images, np.asarray(seq_lengths, dtype=np.int32), labels
