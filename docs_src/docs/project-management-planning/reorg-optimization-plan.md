---
title: Reorg Optimization Plan
sidebar_label: Reorg Optimization
---
# SuperSMM Project Reorganization & Structure Optimization Plan

**Date:** 2025-05-11
**Prepared by:** Cascade AI Agent

---

## 1. Objective
Create a robust, scalable, and maintainable project structure for SuperSMM, following industry SOPs for naming, modularity, logging, debugging, and automation. All changes will be staged in a reversible way to avoid breaking existing workflows.

---

## 2. Analysis Summary
- **Current Structure:**
  - Mixed shell, Python, and JS/React code in `resources/` and `src/`.
  - Some duplication and unclear boundaries between core, UI, and scripts.
  - Logging and debugging logic present but not fully standardized.
  - Asset index now available for all scripts/modules.
- **Key Requirements:**
  - No breakage of existing entrypoints or APIs.
  - Minimize disruption to ongoing research and automation.
  - Enable future-proofing and easy onboarding.

---

## 3. SOP-Based Recommendations

### 3.1. **Directory & System Structure**
- **Top-level:**
  - `/src/` → All core Python packages (core, recognition, preprocessing, utils, config, export, ui, dashboard, etc.)
  - `/scripts/` → All CLI, batch, and utility scripts (Python and shell)
  - `/tests/` → All test code (unit, integration, regression)
  - `/resources/` → Data, models, and large static assets
  - `/docs/` → All documentation, SOPs, and design docs
  - `/logs/`, `/outputs/`, `/exports/`, `/imports/` → For generated files, logs, and results
  - `/dashboard/` or `/src/dashboard/` → All React/JS frontend code
- **Phase1 (Completed):**
  - Archive or refactor `phase1/` after migration

### 3.2. **Naming Conventions**
- **Files/Dirs:**
  - Lowercase, underscores for Python: `symbol_recognizer.py`, `performance_monitor.py`
  - Kebab-case for shell scripts: `lint-and-fix.sh`
  - PascalCase for React components: `Dashboard.jsx`, `MainWindow.py`
- **Classes:** PascalCase (e.g., `OMRPipeline`, `AdvancedPreprocessor`)
- **Functions/Vars:** snake_case (Python), camelCase (JS)
- **Constants:** UPPER_SNAKE_CASE

### 3.3. **Code Structure & Modularity**
- **Core logic** in `src/core/`, helpers in `src/utils/`, ML in `src/recognition/`, config in `src/config/`, UI in `src/ui/`, etc.
- **No business logic in scripts**—scripts should only orchestrate modules.
- **All entrypoints** (`main.py`, `cli.py`, shell scripts) import from `src/`.
- **Tests** import only public APIs.

### 3.4. **Logging & Debugging**
- **Standardize logging:**
  - Use `src/logging_config.py` for all log setup.
  - All modules log to `logs/` with rotation and clear format.
  - Use log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
  - Debug logs go to `debug_logs/` if needed.
- **Error Handling:**
  - Use custom exceptions from `src/core/omr_exceptions.py`.
  - All CLI/scripts should catch and log errors, never fail silently.

### 3.5. **Automation & Indexing**
- **Asset index** (`PROJECT_ASSET_INDEX.md`/`.json`) is kept up to date on every commit and batch repair.
- **Pre-commit hook** ensures no orphan scripts or modules are added.
- **Batch repair scripts** update the index and validate structure.

### 3.6. **Documentation**
- **Maintain:**
  - `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `PROJECT_ASSET_INDEX.md`, and design docs in `/docs/`.
  - SOPs for onboarding, testing, and deployment.
- **Docstrings:**
  - All public functions/classes must have clear docstrings.

### 3.7. **Testing & Validation**
- **Tests in `/tests/`** for all core modules and scripts.
- **CI-ready:** All tests runnable via CLI or GitHub Actions.
- **Validation scripts** (`validate_project.py`) run after major changes.

---

## 4. Migration/Refactor Plan
### 4.1. **Asset Mapping**
The full mapping of all scripts, modules, and assets is now documented in `ASSET_MIGRATION_MAPPING.md` at the project root. This file will be updated as migration proceeds.

### 4.2. **Approval and Progress**
Migration is approved to proceed automatically. Regular updates on progress, validation, and documentation will be provided.

### 4.3. **Staging and Refactoring**
- All moves and refactors will be staged in a dedicated `reorg/structure-2025` branch.
- Move files in batches, updating imports and entrypoints as needed.
- Symlink or stub old paths temporarily to prevent breakage during transition.
- Update scripts (e.g., CLI, shell) to use new import paths.
- Run full test suite and validation after each batch move.
- Update documentation and onboarding guides as structure changes.

---

## 5. Rollback & Safety
- All changes will be staged in a feature branch.
- No files deleted until after successful validation and approval.
- Rollback possible by reverting the branch.

---

## 6. Next Steps
- Review this plan.
- Approve or propose modifications.
- Upon approval, begin staged migration and optimization.

---

**This document is auto-generated and will be updated as the project evolves.**
