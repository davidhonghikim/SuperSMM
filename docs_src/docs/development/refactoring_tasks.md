# SuperSMM Refactoring Tasks

## Current Tasks

### 1. Split `omr_processor.py` (273 lines)

Split into:
1. `src/core/models/ml_symbol_model.py` (91 lines)
   - Class: `MLSymbolModel`
   - Responsibilities: Symbol recognition model management

2. `src/core/processors/pdf_processor.py` (30 lines)
   - Functions: PDF extraction and page handling
   - From: `extract_pdf_pages`

3. `src/core/processors/image_processor.py` (32 lines)
   - Functions: Image preprocessing
   - From: `preprocess_image`

4. `src/core/processors/staff_detector.py` (65 lines)
   - Functions: Staff line detection
   - From: `detect_staff_lines`, `_estimate_staff_line_spacing`

5. `src/core/processors/omr_processor.py` (55 lines)
   - Class: `LocalOMRProcessor`
   - Main orchestration logic
   - Reduced to core workflow management

### Required Changes:

1. Create new directories:
```bash
mkdir -p src/core/models
mkdir -p src/core/processors
```

2. Move and refactor classes:
- Extract `MLSymbolModel` to its own file
- Create focused processor modules
- Update imports and dependencies

3. Update tests:
- Split test files to match new structure
- Add new test cases for extracted functionality

4. Documentation:
- Update module docstrings
- Add architecture documentation
- Update import examples

## Progress Tracking

- [x] Create new directories
  - Created `src/core/models`
  - Created `src/core/processors`
  - Created test directories
- [x] Extract MLSymbolModel
  - Created `ml_symbol_model.py`
  - Added comprehensive tests
- [x] Create PDF processor
  - Created `pdf_processor.py`
  - Focused on single responsibility
- [x] Create image processor
  - Created `image_processor.py`
  - Added error handling
- [x] Create staff detector
  - Created `staff_detector.py`
  - Added staff system grouping
- [x] Refactor main processor
  - Updated `omr_processor.py`
  - Improved orchestration logic
- [x] Update tests
  - Added unit tests for all modules
  - Added integration test for pipeline
- [x] Update documentation
  - Updated README with new architecture
  - Added comprehensive API docs
  - Created architecture guide
- [x] Run all tests and fix failures
  - Fixed image processor error handling
  - All 15 tests passing
- [x] Check code coverage
  - Current coverage: 24%
  - Created coverage improvement plan
  - Identified priority modules for testing
- [ ] Run linting
- [ ] Run type checking

## Quality Checks

For each new module:
- [ ] Line count < 150
- [ ] Single responsibility
- [ ] Clear interfaces
- [ ] Proper error handling
- [ ] Comprehensive tests
- [ ] Complete documentation
- [ ] Check code coverage
- [ ] Run linting
- [ ] Run type checking
