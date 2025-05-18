# OMR Training System Changelog

## [1.3.0] - 2025-05-15
### Added
- Restructured project into a proper Python package (`deep_omr`)
- Added command-line interface for training
- Created configuration loader for YAML-based configuration
- Added wrapper modules for backward compatibility
- Improved documentation with docstrings

### Changed
- Moved source files to a structured directory layout
- Updated import statements to use the new package structure
- Enhanced script files to use the new package structure
- Improved parameterization of training process

## [1.2.0] - 2025-05-15
### Added
- Corpus preparation and validation tools
- Sample dataset creation functionality
- Real-time training monitoring with live dashboard
- Test training script for quick verification
- Production training script with optimized settings

### Fixed
- Path resolution in corpus handling
- Training interruption due to missing images

## [1.1.0] - 2025-05-10
### Added
- Auto-recovery training system
- System resource monitoring
- Email notifications for training events
- TensorFlow optimizations for improved performance

### Changed
- Improved checkpoint management
- Enhanced error handling and logging

## [1.0.0] - 2025-05-01
### Added
- Initial release of OMR training system
- Basic training script
- Support for semantic model training
- Checkpoint saving and loading
