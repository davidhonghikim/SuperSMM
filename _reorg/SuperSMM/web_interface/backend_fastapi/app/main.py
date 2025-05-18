# web_interface/backend_fastapi/app/main.py
from fastapi import FastAPI
from .routers import documentation, debug_symbols, logs
import asyncio # Added for shutdown event

app = FastAPI(
    title="SuperSMM Interface API",
    description="API for interacting with SuperSMM project systems (docs, logs, debug symbols).",
    version="0.1.0"
)

app.include_router(documentation.router, prefix="/api")
app.include_router(debug_symbols.router, prefix="/api")
app.include_router(logs.router, prefix="/api") # Include /api prefix

@app.get("/")
async def root():
    return {"message": "Welcome to the SuperSMM Web Interface API"}

# Placeholder for future routers
# from .routers import documentation, debug_symbols
# app.include_router(documentation.router)
# app.include_router(debug_symbols.router)

# For now, a simple health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down server...")
    # Create a list of close tasks to run concurrently
    close_tasks = [conn.close() for conn in logs.active_connections]
    if close_tasks:
        await asyncio.gather(*close_tasks, return_exceptions=True) # return_exceptions to log errors but not stop others
        print(f"Closed {len(close_tasks)} active WebSocket connections.")
    else:
        print("No active WebSocket connections to close.")
    logs.active_connections.clear() # Clear the set after closing
    print("Server shutdown complete.")
