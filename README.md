# SuperSMM - Sheet Music Master

[![CI/CD](https://github.com/davidhonghikim/SuperSMM/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/davidhonghikim/SuperSMM/actions/workflows/ci-cd.yml)
[![Docker Build](https://img.shields.io/github/actions/workflow/status/davidhonghikim/SuperSMM/docker-build.yml?label=docker)](https://github.com/davidhonghikim/SuperSMM/actions/workflows/docker-build.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
SuperSMM is an advanced Optical Music Recognition (OMR) system that converts sheet music images into machine-readable music notation. It features a modular architecture, comprehensive test coverage, and supports both traditional and machine learning-based processing pipelines.

## Package Structure

```
src/
├── supersmm/                      # Main package
│   ├── core/                      # Core OMR functionality
│   │   ├── pipeline/             # Pipeline components
│   │   ├── models/               # Core model definitions
│   │   └── utils/                # Core utilities
│   │
│   ├── preprocessing/         # Image preprocessing
│   │   ├── staff_removal/        # Staff line detection/removal
│   │   ├── normalization/        # Image normalization
│   │   └── augmentation/         # Data augmentation
│   │
│   ├── recognition/          # Symbol recognition
│   │   ├── ctc/                  # CTC-based recognition
│   │   ├── hmm/                  # HMM-based recognition
│   │   └── postprocessing/       # Recognition post-processing
│   │
│   ├── segmentation/         # Music symbol segmentation
│   │   ├── staff/               # Staff detection
│   │   ├── symbol/              # Symbol extraction
│   │   └── grouping/            # Symbol grouping
│   │
│   ├── export/               # Export functionality
│   │   ├── musicxml/            # MusicXML export
│   │   └── midi/                # MIDI export
│   │
│   └── utils/                # Shared utilities
│       ├── logging/             # Logging configuration
│       ├── visualization/       # Visualization tools
│       └── helpers/             # Helper functions
│
├── tests/                     # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures
│
└── scripts/                   # Utility scripts
    ├── train.py               # Training script
    ├── predict.py             # Prediction script
    └── evaluate.py           # Evaluation script
```

## Key Components

### Core Pipeline
- **OMR Pipeline**: Main orchestration of the recognition process
- **Configuration Management**: Centralized configuration handling
- **Error Handling**: Robust error handling and recovery

### Preprocessing
- **Staff Removal**: Advanced staff line detection and removal
- **Image Enhancement**: Noise reduction, binarization, and normalization
- **Augmentation**: Data augmentation for training

### Recognition
- **CTC-based Recognition**: Deep learning-based symbol recognition
- **HMM Decoding**: Probabilistic sequence modeling
- **Post-processing**: Error correction and refinement

### Export
- **MusicXML Export**: Standard music notation format
- **MIDI Export**: Playable music format

## Features
- **Modular Architecture**: Easily extensible components
- **High Accuracy**: Advanced ML-based recognition
- **Batch Processing**: Efficient handling of multiple documents
- **Comprehensive Testing**: Extensive test coverage
- **Containerized**: Easy deployment with Docker

## Prerequisites
- Python 3.8, 3.9, or 3.10
- JDK 21
- Tesseract OCR 5.x
- Docker (optional, for containerized deployment)

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/davidhonghikim/SheetMasterMusic.git
cd SuperSMM

# Build and run the development container
docker build -t supersmm:dev --target dev .
docker run -it --rm -v $(pwd):/app supersmm:dev

# Or for production
docker build -t supersmm:latest --target prod .
docker run -p 8000:8000 supersmm:latest
```

### Local Development

#### 1. Clone the Repository
```bash
git clone https://github.com/davidhonghikim/SheetMasterMusic.git
cd SuperSMM
```

#### 2. Set up Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install package in development mode with all dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

#### 3. Run Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests

# Run with coverage report
pytest --cov=supersmm --cov-report=term-missing
```

## Development Workflow

### Code Style
- We use `black` for code formatting and `isort` for import sorting
- Pre-commit hooks are configured to enforce code style
- Run `black .` and `isort .` to format code

### Type Checking
- Static type checking is done using `mypy`
- Run `mypy src/` to check for type errors

### Linting
- `flake8` is used for code linting
- Run `flake8 src/` to check for style issues

### Testing
- Write unit tests in `tests/unit/`
- Write integration tests in `tests/integration/`
- Use fixtures from `tests/fixtures/` for test data

### Documentation
- Docstrings follow Google style guide
- Run `sphinx-build docs/ docs/_build/` to build documentation

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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- [ ] Custom model training pipeline

## Project Structure

