# web_interface/backend_fastapi/app/services/log_service.py
import asyncio
from pathlib import Path
from typing import List, Dict, AsyncGenerator

# Define the root directory for logs, relative to the project root
# Assumes this service file is at SuperSMM/web_interface/backend_fastapi/app/services/log_service.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOGS_DIRECTORY = PROJECT_ROOT / "logs"

class LogService:
    def __init__(self):
        self.logs_dir = LOGS_DIRECTORY
        if not self.logs_dir.exists():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created logs directory at: {self.logs_dir}") # Replace with actual logging

    def list_log_files(self) -> List[Dict[str, str]]:
        """Lists available log files with their names and last modified times."""
        log_files = []
        if not self.logs_dir.is_dir():
            return log_files

        for log_file in self.logs_dir.glob("*.log"):
            if log_file.is_file():
                try:
                    stat = log_file.stat()
                    log_files.append({
                        "name": log_file.name,
                        "path": str(log_file),
                        "last_modified": stat.st_mtime, # timestamp
                        "size_bytes": stat.st_size
                    })
                except Exception as e:
                    print(f"Error stating file {log_file.name}: {e}") # Replace with actual logging
        
        # Sort by last modified, newest first
        log_files.sort(key=lambda x: x["last_modified"], reverse=True)
        return log_files

    def get_log_file_path(self, log_file_name: str) -> Path:
        """Validates and returns the full path to a log file."""
        log_file_path = (self.logs_dir / log_file_name).resolve()
        # Security check: ensure the path is still within the logs_dir
        if not log_file_path.exists() or not log_file_path.is_file() or self.logs_dir not in log_file_path.parents:
             # A more robust check for Python 3.9+ would be: log_file_path.is_relative_to(self.logs_dir)
            raise FileNotFoundError(f"Log file '{log_file_name}' not found or access denied.")
        return log_file_path

    async def stream_log_file(self, log_file_path: Path, initial_lines: int = 50) -> AsyncGenerator[str, None]:
        """
        Streams a log file. Sends the last N lines initially, then tails the file for new lines.
        This implementation will send lines as they appear.
        """
        # Send initial last N lines
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                start_index = max(0, len(lines) - initial_lines)
                for i in range(start_index, len(lines)):
                    yield lines[i]
                
                # Seek to the end of the file to start tailing
                f.seek(0, 2) # 0 offset from whence=2 (end of file)
                
                while True:
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.1)  # Wait for new lines
                        continue
                    yield line
        except FileNotFoundError:
            yield f"ERROR: Log file {log_file_path.name} not found.\n"
        except Exception as e:
            yield f"ERROR: Could not read log file {log_file_path.name}: {e}\n"

# Example Usage (for direct testing)
if __name__ == "__main__":
    service = LogService()
    print("--- Available Log Files ---")
    files = service.list_log_files()
    for f_info in files:
        print(f_info)

    async def main_test_stream():
        if files:
            print(f"\n--- Streaming {files[0]['name']} (last 10 lines then tailing) ---")
            test_log_path = service.get_log_file_path(files[0]['name'])
            # Create or append to the test log file for streaming
            with open(test_log_path, 'a') as tf:
                tf.write("Initial line for testing stream.\n")
            
            count = 0
            async for line_content in service.stream_log_file(test_log_path, initial_lines=10):
                print(line_content, end='')
                count += 1
                if count > 15 and "Initial line" in line_content: # Simulate some streaming then stop for test
                    # In a real scenario, another process would append to the log file
                    with open(test_log_path, 'a') as tf:
                        tf.write(f"Another streamed line {count}\n")
                if count > 20: # Stop after a few lines for testing
                    print("\nStopping test stream.")
                    break
        else:
            print("No log files to test streaming.")

    # asyncio.run(main_test_stream()) # Commented out to prevent auto-run issues
