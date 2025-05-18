# Core API Documentation

## Models

### MLSymbolModel

```python
from src.core.models.ml_symbol_model import MLSymbolModel
```

Machine learning model for recognizing musical symbols in sheet music.

#### Methods

##### `__init__(model_path='resources/ml_models/symbol_recognition')`
Initialize ML Symbol Recognition Model.

- **Args:**
  - `model_path (str)`: Path to saved TensorFlow model

##### `predict_symbols(symbol_images: List[np.ndarray]) -> List[Dict[str, Any]]`
Predict musical symbols from preprocessed images.

- **Args:**
  - `symbol_images (List[np.ndarray])`: List of preprocessed symbol candidate images
- **Returns:**
  - List of dictionaries containing:
    - `label`: Predicted symbol label
    - `confidence`: Prediction confidence score

## Processors

### PDFProcessor

```python
from src.core.processors.pdf_processor import extract_pdf_pages
```

Functions for handling PDF files and page extraction.

#### Functions

##### `extract_pdf_pages(pdf_path: str) -> List[np.ndarray]`
Extract pages from PDF and convert to images.

- **Args:**
  - `pdf_path (str)`: Path to PDF file
- **Returns:**
  - List of page images as numpy arrays
- **Raises:**
  - `FileNotFoundError`: If PDF file does not exist
  - `ValueError`: If PDF cannot be processed

### ImageProcessor

```python
from src.core.processors.image_processor import preprocess_image
```

Image preprocessing utilities for OMR.

#### Functions

##### `preprocess_image(image: np.ndarray) -> Dict[str, Any]`
Advanced image preprocessing for OMR.

- **Args:**
  - `image (np.ndarray)`: Input image
- **Returns:**
  - Dictionary containing:
    - `grayscale`: Grayscale image
    - `denoised`: Denoised image
    - `binary`: Final binary image
    - `threshold_params`: Parameters used for thresholding

### StaffDetector

```python
from src.core.processors.staff_detector import detect_staff_lines
```

Staff line detection utilities.

#### Functions

##### `detect_staff_lines(binary_image: np.ndarray) -> Dict[str, Any]`
Detect staff lines in sheet music.

- **Args:**
  - `binary_image (np.ndarray)`: Preprocessed binary image
- **Returns:**
  - Dictionary containing:
    - `total_lines`: Total number of detected lines
    - `horizontal_lines`: Number of horizontal lines
    - `staff_line_spacing`: Average spacing between staff lines
    - `staff_systems`: List of detected staff systems
    - `line_positions`: List of line y-positions

### LocalOMRProcessor

```python
from src.core.processors.omr_processor import LocalOMRProcessor
```

Main orchestrator for local OMR processing.

#### Methods

##### `__init__()`
Initialize the OMR processor with required components.

##### `process_sheet_music(pdf_path: str) -> List[Dict[str, Any]]`
Process entire sheet music PDF.

- **Args:**
  - `pdf_path (str)`: Path to PDF file
- **Returns:**
  - List of dictionaries containing:
    - `pdf_path`: Original PDF path
    - `page_number`: Page index
    - `preprocessed_image`: Dict with preprocessing results
    - `staff_line_detection`: Dict with staff line info
    - `symbol_labels`: List of detected symbols

## Usage Examples

### Basic Usage

```python
from src.core.processors.omr_processor import LocalOMRProcessor

# Initialize processor
processor = LocalOMRProcessor()

# Process a sheet music PDF
results = processor.process_sheet_music('path/to/sheet_music.pdf')

# Access results
for page in results:
    print(f"Page {page['page_number']}")
    print(f"Found {len(page['symbol_labels'])} symbols")
    print(f"Staff lines: {page['staff_line_detection']['total_lines']}")
```

### Advanced Usage

```python
import numpy as np
from src.core.processors.image_processor import preprocess_image
from src.core.processors.staff_detector import detect_staff_lines
from src.core.models.ml_symbol_model import MLSymbolModel

# Load and preprocess image
image = ... # Load your image
preprocessed = preprocess_image(image)

# Detect staff lines
staff_info = detect_staff_lines(preprocessed['binary'])

# Initialize model and predict symbols
model = MLSymbolModel()
symbols = model.predict_symbols([...])  # Your symbol candidates
```
