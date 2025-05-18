# Dependency Cleanup Plan

## Current Issues

1. **Version Conflicts**
   - Different minimum versions between main project and TF-Deep-OMR
   - Wide version ranges in requirements

2. **Redundant Dependencies**
   - Multiple image processing libraries (OpenCV, Pillow)
   - Multiple visualization libraries (Matplotlib, Seaborn)
   - Development tools mixed with production dependencies

3. **Dependency Management**
   - No clear separation between dev and prod dependencies
   - No lock files for reproducible builds
   - No centralized configuration for development tools

## Proposed Changes

### 1. Directory Structure
```
requirements/
├── dev.txt       # Development dependencies
├── prod.txt      # Production dependencies
└── constraints.txt  # Version constraints (optional)
```

### 2. Dependency Management

#### Production Dependencies (`requirements/prod.txt`)
- Core runtime dependencies only
- Pinned to minimum working versions
- No development or testing tools

#### Development Dependencies (`requirements/dev.txt`)
- Includes all development tools
- References prod.txt to ensure consistency
- Includes testing and documentation tools

#### Setup Configuration (`setup.cfg`)
- Centralized package configuration
- Defines optional dependencies (extras_require)
- Configures development tools (flake8, mypy, isort)

### 3. Version Pinning Strategy

#### Production Dependencies
- Use `>=` for minimum versions
- Avoid upper bounds unless necessary
- Document known working versions

#### Development Dependencies
- Can be more flexible
- Use `>=` for minimum versions
- Consider pinning exact versions in CI

### 4. Migration Steps

1. **Phase 1: Setup New Structure**
   - Create new requirements directory
   - Move production dependencies to `requirements/prod.txt`
   - Create `requirements/dev.txt` with development tools
   - Add `setup.cfg` with package metadata and tool configurations

2. **Phase 2: Update Documentation**
   - Update README with new installation instructions
   - Document development setup
   - Add contributing guidelines

3. **Phase 3: CI/CD Updates**
   - Update CI/CD pipelines to use new requirements
   - Add dependency caching
   - Add dependency update automation

4. **Phase 4: Cleanup**
   - Remove old requirements files
   - Update Dockerfiles and deployment scripts
   - Verify all tests pass with new dependencies

### 5. Recommended Tools

#### For Development
```bash
# Install in development mode with all extras
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Lint code
flake8
```

#### For Production
```bash
# Install production dependencies only
pip install -r requirements/prod.txt

# Or install package
pip install .
```

### 6. Future Improvements

1. **Dependency Updates**
   - Add dependabot for automated updates
   - Schedule regular dependency reviews

2. **Lock Files**
   - Consider adding `pip-tools` for deterministic builds
   - Generate `requirements.txt` from `setup.cfg`

3. **Containerization**
   - Create multi-stage Dockerfiles
   - Separate build and runtime dependencies

4. **Documentation**
   - Add dependency graph visualization
   - Document version compatibility
   - Add upgrade guides

## Rollback Plan

1. Keep the old `requirements.txt` during transition
2. Tag the current working state
3. Document the rollback procedure

---

*Last updated: May 17, 2025*
