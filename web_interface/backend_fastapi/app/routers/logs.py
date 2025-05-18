# web_interface/backend_fastapi/app/routers/logs.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import List, Dict, Set

from ..services.log_service import LogService

# Set to keep track of active WebSocket connections
active_connections: Set[WebSocket] = set()

router = APIRouter(
    prefix="/logs",
    tags=["Logging"],
)

log_service = LogService()

@router.get("/files", response_model=List[Dict])
async def get_log_files_list():
    """
    Retrieve a list of available log files in the designated logs directory.
    """
    try:
        return log_service.list_log_files()
    except Exception as e:
        # Log the exception e
        raise HTTPException(status_code=500, detail=f"Failed to list log files: {str(e)}")

@router.websocket("/stream/{log_file_name}")
async def websocket_log_stream(websocket: WebSocket, log_file_name: str, lines: int = Query(50, description="Number of initial lines to send from the end of the file.")):
    """
    WebSocket endpoint to stream a log file.
    Sends the last N lines first, then tails the file for new lines.
    `log_file_name` should be the name of the file, e.g., 'app.log'.
    `lines` query parameter controls how many recent lines are sent upon connection.
    """
    await websocket.accept()
    active_connections.add(websocket)
    print(f"WebSocket connected: {log_file_name}. Total connections: {len(active_connections)}")
    try:
        log_path = log_service.get_log_file_path(log_file_name)
        async for line_content in log_service.stream_log_file(log_path, initial_lines=lines):
            await websocket.send_text(line_content)
    except FileNotFoundError:
        await websocket.send_text(f"ERROR: Log file '{log_file_name}' not found.\n")
        await websocket.close(code=1008) # Policy Violation or similar, file not found
    except WebSocketDisconnect:
        print(f"Client disconnected from log stream: {log_file_name}")
    except Exception as e:
        # Log the exception properly in a real app
        error_message = f"ERROR: An error occurred while streaming log '{log_file_name}': {str(e)}\n"
        print(error_message) # For server console
        try:
            await websocket.send_text(error_message)
        except Exception as send_err:
            print(f"Error sending error message to client: {send_err}")
        await websocket.close(code=1011) # Internal Error
    finally:
        # Ensure the WebSocket is closed if not already
        # This might be redundant if already closed in except blocks, but good for cleanup
        if websocket.client_state != websocket.client_state.DISCONNECTED:
            # Check if reason can be added here based on FastAPI version
            await websocket.close()
            print(f"Log stream WebSocket for {log_file_name} closed programmatically.")
        active_connections.remove(websocket)
        print(f"WebSocket disconnected: {log_file_name}. Remaining connections: {len(active_connections)}")

