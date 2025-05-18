---
title: Logging Standard
sidebar_label: Logging
---
# SuperSMM Logging Standard

## 1. Purpose

This standard defines a consistent approach to logging across all SuperSMM services and applications. The goals are to:
- Facilitate efficient debugging and troubleshooting.
- Enable effective monitoring and alerting.
- Ensure logs are easily parsable by humans, scripts, and log analysis tools (including AI agents).
- Maintain an organized and manageable log file structure.

## 2. Log Format: Structured JSON

All log entries MUST be formatted as a single line of JSON. This allows for easy parsing and querying.

**Core Fields:**

| Field Name  | Type   | Description                                                                                                  | Example                               | Requirement |
|-------------|--------|--------------------------------------------------------------------------------------------------------------|---------------------------------------|-------------|
| `timestamp` | String | ISO 8601 timestamp with timezone (UTC preferred). Millisecond precision recommended.                         | `"2025-05-12T21:10:05.543Z"`          | **Required**|
| `level`     | String | Log severity level. Allowed values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.                         | `"INFO"`                              | **Required**|
| `service`   | String | Name of the application or major component generating the log.                                               | `"fastapi_backend"`, `"omr_pipeline"` | **Required**|
| `module`    | String | The specific module/file where the log originated (e.g., Python's `__name__`).                             | `"app.services.log_service"`        | **Required**|
| `message`   | String | The primary, human-readable log message. Should be concise and informative.                                | `"User logged in successfully"`         | **Required**|
| `function`  | String | The function or method name where the log occurred.                                                          | `"authenticate_user"`                 | Recommended |
| `context`   | Object | Key-value pairs providing additional structured context relevant to the event. Avoid sensitive data here.  | `{"request_id": "xyz789", "user_id": 123}` | Optional    |
| `exception` | Object | Error details if `level` is `ERROR` or `CRITICAL`. Should include type, message, and stack trace (string). | `{"type": "ValueError", ...}`       | Optional    |
| `trace_id`  | String | A unique identifier to correlate log messages belonging to the same request or transaction across services.  | `"a1b2c3d4-e5f6-..."`              | Optional    |


**Example Log Entries:**

```json
{"timestamp": "2025-05-12T21:10:05.543Z", "level": "INFO", "service": "fastapi_backend", "module": "app.routers.auth", "function": "login", "message": "User login successful", "context": {"user_id": 123, "client_ip": "192.168.1.100"}, "trace_id": "req-778"}
```

```json
{"timestamp": "2025-05-12T21:11:15.987Z", "level": "ERROR", "service": "omr_pipeline", "module": "segmentation.staff_detector", "function": "find_staff_lines", "message": "Failed to process image", "context": {"image_path": "/tmp/img_abc.png"}, "exception": {"type": "ValueError", "message": "Unsupported image depth", "traceback": "Traceback (most recent call last):\n  File \"..."..."}, "trace_id": "proc-456"}
```

## 3. Log Levels

Use the following levels appropriately:

- **`DEBUG`**: Verbose details for development/diagnosis only. (e.g., variable states, entry/exit points). *Should be disabled in production by default.*
- **`INFO`**: Routine operational information. (e.g., service started, request completed, configuration loaded).
- **`WARNING`**: Potential issue or unusual event, but operation continues. (e.g., fallback used, retry occurred, deprecated feature accessed).
- **`ERROR`**: Serious error preventing a specific task/operation from completing. Requires attention. (e.g., failed database write, invalid input causing failure, exception caught).
- **`CRITICAL`**: Severe error potentially impacting overall service stability or causing shutdown. Requires immediate attention. (e.g., database connection lost, unrecoverable error).

## 4. File Structure and Rotation

- **Base Directory:** `logs/` at the project root (`/Users/danger/CascadeProjects/LOO/SuperSMM/logs/`).
- **Service Subdirectories:** Logs are segregated by the `service` field into subdirectories (e.g., `logs/fastapi_backend/`, `logs/omr_pipeline/`, `logs/scripts/`).
- **Log Filename Format:** Daily log files named `YYYY-MM-DD.jsonl`.
  - Example: `logs/fastapi_backend/2025-05-12.jsonl`
- **Rotation:**
    - Logs rotate daily (at midnight UTC).
    - Rotated logs should be compressed (e.g., `.gz`).
    - Configure retention period (e.g., keep logs for 30 days).
- **Permissions:** Log files should have appropriate permissions to prevent unauthorized access, especially if they might contain contextual data.

## 5. Implementation Guidelines

### Python (FastAPI / General)
- Use the standard `logging` library.
- Configure logging centrally (e.g., in `app/core/logging_config.py` or similar).
- Use `logging.handlers.TimedRotatingFileHandler` for rotation and `logging.Formatter` (or a JSON formatter like `python-json-logger`) to produce the standard JSON format.
- Obtain loggers per module: `logger = logging.getLogger(__name__)`.
- Include contextual information using `extra` dictionary: `logger.info("User logged in", extra={"context": {"user_id": 123}, "trace_id": "req-778"})`.
- Configure root logger and handlers appropriately (e.g., level, propagation).

### Frontend (React)
- Use `console.log()`, `console.warn()`, `console.error()` for development debugging.
- To capture significant frontend events/errors in persistent logs:
    - Implement a dedicated backend API endpoint (e.g., `/api/log_event`).
    - Frontend sends structured JSON log data (matching the standard format where possible) to this endpoint.
    - The backend endpoint validates the data and uses its standard logger to write the entry to a dedicated service log file (e.g., `logs/frontend/YYYY-MM-DD.jsonl`).

## 6. Viewing and Querying Logs

- The structured JSON format makes logs suitable for command-line tools like `jq` or dedicated log aggregation/analysis platforms (e.g., ELK stack, Splunk, Datadog, Grafana Loki).
- Example `jq` query to find all ERROR messages from the backend service:
  ```bash
  jq 'select(.level == "ERROR" and .service == "fastapi_backend")' logs/fastapi_backend/*.jsonl
  ```
