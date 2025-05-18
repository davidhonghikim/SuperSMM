# SuperSMM Architecture Guide

## Overview

SuperSMM follows a modular architecture designed for maintainability, testability, and extensibility. This guide explains the core architectural concepts and how to work with them.

## Core Principles

1. **Modularity**: Each component has a single responsibility
2. **Testability**: All components are designed for easy testing
3. **Extensibility**: New features can be added without modifying existing code
4. **Error Handling**: Comprehensive error handling and logging
5. **Documentation**: Clear API documentation and usage examples

## Component Architecture

### 1. Models Layer (`src/core/models/`)

The models layer handles all machine learning and prediction tasks.

```
models/
└── ml_symbol_model.py   # Symbol recognition model
```

Key features:
- Model loading and initialization
- Symbol prediction
- Mock model for testing
- Error handling for missing models

### 2. Processors Layer (`src/core/processors/`)

The processors layer contains all data processing components.

```
processors/
├── pdf_processor.py     # PDF handling
├── image_processor.py   # Image preprocessing
├── staff_detector.py    # Staff line detection
└── omr_processor.py     # Main orchestration
```

Features:
- Independent, focused processors
- Clear input/output contracts
- Comprehensive error handling
- Progress logging

### 3. Utilities Layer (`src/utils/`)

Common utilities and helper functions.

```
utils/
├── logging_utils.py     # Logging setup
├── debug_symbols.py     # Debug helpers
└── error_handler.py     # Error handling
```

## Data Flow

1. **Input**: PDF file → `pdf_processor.py`
2. **Preprocessing**: Raw images → `image_processor.py`
3. **Analysis**: 
   - Binary images → `staff_detector.py`
   - Symbol candidates → `ml_symbol_model.py`
4. **Orchestration**: All components → `omr_processor.py`

## Error Handling

### Hierarchy

1. **Component-level errors**: Handled within each processor
2. **Pipeline errors**: Handled by OMR processor
3. **System errors**: Handled by error handler utility

### Error Types

```python
# Example error handling
try:
    result = process_component()
except ComponentError as e:
    # Log and return safe default
    logger.error(f"Component error: {e}")
    return default_value
except Exception as e:
    # Log and propagate system errors
    logger.critical(f"System error: {e}")
    raise
```

## Testing Strategy

### Unit Tests

```
tests/core/
├── models/
│   └── test_ml_symbol_model.py
└── processors/
    ├── test_pdf_processor.py
    ├── test_image_processor.py
    ├── test_staff_detector.py
    └── test_omr_processor.py
```

Test types:
1. Function-level unit tests
2. Component integration tests
3. Mock-based isolation tests
4. Error case testing

### Test Data

```
tests/fixtures/
├── pdfs/          # Test PDF files
├── images/        # Test images
└── expected/      # Expected outputs
```

## Logging

### Log Levels

1. **DEBUG**: Detailed processing information
2. **INFO**: Progress and success messages
3. **WARNING**: Non-critical issues
4. **ERROR**: Component failures
5. **CRITICAL**: System failures

### Log Structure

```
logs/
├── app/           # Application logs
├── ml/            # ML processing logs
└── debug/         # Debug information
```

## Adding New Features

1. **Plan**:
   - Identify the component layer
   - Define input/output contract
   - Plan error handling

2. **Implement**:
   - Create new module
   - Follow existing patterns
   - Add comprehensive tests

3. **Document**:
   - Update API documentation
   - Add usage examples
   - Update architecture docs

## Best Practices

1. **Code Organization**:
   - Keep files under 150 lines
   - One class/responsibility per file
   - Clear file and function names

2. **Documentation**:
   - Docstrings for all public APIs
   - Type hints for all functions
   - Usage examples in docs

3. **Testing**:
   - Test all error cases
   - Use appropriate fixtures
   - Mock external dependencies

4. **Logging**:
   - Log all important operations
   - Use appropriate log levels
   - Include context in messages

## Example: Adding a New Processor

```python
# src/core/processors/new_processor.py

import logging
from typing import Dict, Any

logger = logging.getLogger('new_processor')

def process_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process input data with new functionality.
    
    Args:
        input_data: Input dictionary
        
    Returns:
        Processed results
        
    Raises:
        ValueError: If input is invalid
    """
    try:
        # Process data
        result = {'status': 'success'}
        logger.info('Processing completed')
        return result
    except Exception as e:
        logger.error(f'Processing failed: {e}')
        raise
```

## Deployment

1. **Testing**:
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific component tests
   pytest tests/core/processors/
   ```

2. **Documentation**:
   ```bash
   # Build documentation
   cd docs_src && npm run build
   ```

3. **Deployment**:
   ```bash
   # Build application
   ./build.sh
   
   # Deploy
   ./deploy.sh
   ```
