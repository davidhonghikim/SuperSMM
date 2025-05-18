# SuperSMM - Sheet Music Master

## Project Overview
A hybrid PDF-to-MusicXML converter with intelligent processing capabilities, featuring modular design and comprehensive test coverage.

## Architecture

### Core Components
- **Models**: ML-based symbol recognition
  - `ml_symbol_model.py`: TensorFlow-based music symbol classifier

### Processors
- **PDF Processing**: `pdf_processor.py`
  - Extracts and converts PDF pages to images
  - Handles multi-page documents

- **Image Processing**: `image_processor.py`
  - Advanced preprocessing pipeline
  - Noise reduction and binarization

- **Staff Detection**: `staff_detector.py`
  - Staff line detection using Hough Transform
  - Staff system grouping

- **OMR Pipeline**: `omr_processor.py`
  - Main orchestration logic
  - Coordinates processing workflow

## Features
- Local OMR (Optical Music Recognition) processing
- Intelligent PDF page extraction
- Advanced staff line detection
- ML-based symbol recognition
- Modular, testable architecture

## Prerequisites
- Python 3.11+
- JDK 21
- Tesseract OCR 5.x

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/davidhonghikim/SheetMasterMusic.git
cd SuperSMM
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ML model dependencies (if using GPU)
pip install -r requirements-gpu.txt  # Optional
```

### 4. Run Tests
```bash
# Run all tests
PYTHONPATH=. pytest tests/

# Run specific test modules
pytest tests/core/models/  # Test ML models
pytest tests/core/processors/  # Test processors
```

### 5. Launch Application
```bash
# Start the application
./launch_app.sh

# Start with debug logging
DEBUG=1 ./launch_app.sh
```

## Development Roadmap

### Completed
- [x] Modular code architecture
- [x] Comprehensive test suite
- [x] Staff line detection
- [x] Basic symbol recognition

### In Progress
- [ ] Improve OMR accuracy with advanced ML models
- [ ] Add cloud processing capabilities
- [ ] Implement MusicXML export
- [ ] Create mobile/web interfaces

### Future Plans
- [ ] GPU acceleration support
- [ ] Batch processing capabilities
- [ ] Real-time recognition mode
- [ ] Custom model training pipeline

## Project Structure

```
src/
├── core/
│   ├── models/          # ML models
│   └── processors/      # Processing pipeline
├── utils/              # Shared utilities
└── web_interface/      # Frontend components

tests/
├── core/
│   ├── models/         # Model tests
│   └── processors/     # Processor tests
└── utils/             # Utility tests

logs/                  # Application logs
├── app/               # Main app logs
├── ml/                # ML processing logs
└── debug/            # Debug information
```

## Contributing
Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
