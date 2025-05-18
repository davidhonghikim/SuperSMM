# web_interface/backend_fastapi/app/routers/debug_symbols.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from ..services.debug_symbol_service import DebugSymbolService

router = APIRouter(
    prefix="/debug",
    tags=["Debug Symbols"],
)

debug_service = DebugSymbolService()

@router.get("/symbols", response_model=List[Dict[str, Any]])
async def get_all_debug_symbols():
    """
    Retrieve a list of all debug symbols (code, message_template, default_color)
    parsed from the `src/utils/debug_symbols.py` file.
    """
    symbols = debug_service.get_debug_symbols()
    # if not symbols: # Depending on if an empty list is an error or valid
    #     raise HTTPException(status_code=404, detail="No debug symbols found or error parsing file.")
    return symbols
