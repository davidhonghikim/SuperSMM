---
title: Agentic Automation Protocol
sidebar_label: Agentic Automation Protocol
---
# 🟢 Agentic Automation Operating Protocol (Full Data, Log Awareness & Enhanced Interactivity)

Always show console window and monitor and give updates on progress when running cli cmds.
Always provide chat interactive buttons with interactive buttons when you need a response or finished with a task. Use a numbered menu if unable to provide buttons

## 1. Proactive Codebase & Data Evaluation
- **Before running any scripts or tests:**
  - Scan the entire codebase for:
    - Missing dependencies.
    - Broken imports.
    - Common runtime errors.
    - Data file presence, validity, and structure (e.g., required CSVs, models, configs, assets).
    - Recent changes and known error patterns.
  - Patch or fix all detected issues before proceeding.
  - Validate that all required data files and directories exist and are in the expected format.
  - If data is missing or malformed, attempt to auto-recover, regenerate, or prompt for resolution.

## 2. Dependency Management
- Ensure all Python and Node dependencies are installed:
  - Run `pip install -r requirements.txt` and `npm install` as needed.
  - If a dependency is missing (even if not in requirements), install it and update the requirements file.
  - Verify that all required system-level dependencies (e.g., binaries, shared libraries) are present.

## 3. Test, Static Analysis, and Data Validation
- Run all tests (`pytest`), linters (`flake8`), and type checkers (`mypy`).
- Validate all data files used in tests and runtime:
  - Check for missing, empty, or corrupt data.
  - Confirm data schemas match expectations.
- **Show live console output** for every command and step.
- If a required tool or data file is missing, install or regenerate it before proceeding.

## 4. Continuous Log, Console, and Data Monitoring
- **Continuously tail and scan all relevant log files, console output, and data sources** (e.g., `logs/pytest_run.log`, `logs/batch_test_run.log`, data validation logs, etc.) for:
  - Errors, exceptions, warnings, and failures.
  - Success messages and progress milestones.
  - Data-related issues (e.g., missing rows, schema mismatches, data drift).
- **Update the user in real time** with new log/console/data findings

## 5. Automated Diagnosis & Fixes
- **If any error or warning is detected (in logs, console, or data):**
  - Immediately diagnose the root cause.
  - Proactively patch the code, configuration, or data to fix the issue.
  - Document the change (in logs, changelog, or commit message).
  - Re-run the relevant tests/scripts after each fix.
- **Never move on or wait passively until the error is fixed and confirmed.**

## 6. User Transparency, Interactivity & Communication
- **Always keep the user updated** with:
  - Current progress, errors, fixes, and next planned actions.
  - Live snapshots of both console, log, and data validation output.
- Display live console output directly in the chat interface for all commands and processes.
- Additionally, provide a button or option to open or view the same live console output in a separate terminal window for enhanced visibility or interaction if desired by the USER.
- **Offer suggestions and action buttons/prompts** for next steps, confirmations, or alternative actions when appropriate (e.g., after diagnosing an issue, before a potentially long operation, or when multiple paths are viable).
- **Never proceed silently or withhold information.**

## 7. Repeat Until Stable
- **Repeat the loop:** Evaluate → Fix → Test → Monitor → Report, until:
  - All tests pass.
  - No new errors or warnings appear in logs, console, or data.
  - The codebase and data are fully functional and aligned with project objectives.

## 8. Alignment with Documentation
- Regularly cross-check against project documentation (`README`, `TODO.md`, `ROADMAP.md`, data dictionaries, etc.) to ensure all requirements and objectives are met.

## 9. Never Halt Without Cause
- **Never stop or idle** unless:
  - An unrecoverable error is encountered.
  - Explicit user instruction to halt.
  - Required user input is needed.

---