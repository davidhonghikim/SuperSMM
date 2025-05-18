# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial project structure for OMR pipeline.
- `ConfigManager` for handling configurations from defaults, YAML, and environment variables.
- `pdf_processor.py` for extracting and converting PDF pages to images.
- `advanced_preprocessor.py` for detailed image preprocessing including normalization, binarization, and staff line removal (with HMM and fallback).
- Initial `ROADMAP.md` and `TODO.md` for project tracking.
- `docs_src/docs/memories/` directory for Docusaurus AI context.

### Changed
- Refactored `pdf_processor.py` to use helper functions for page index determination and image conversion.
- Refactored `advanced_preprocessor.py` into multiple stages for clarity and robustness, and to include `save_intermediate_stages` config.

### Fixed
- (Placeholder for fixes)

