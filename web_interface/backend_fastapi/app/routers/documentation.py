# web_interface/backend_fastapi/app/routers/documentation.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional

from ..services.documentation_service import DocumentationService

router = APIRouter(
    prefix="/documentation",
    tags=["Documentation"],
)

# Initialize services
# In a more complex app, you might use dependency injection (e.g., FastAPI's Depends)
doc_service = DocumentationService()

@router.get("/references", response_model=List[Dict[str, str]])
async def get_all_documentation_references():
    """
    Retrieve a list of all documentation references (DOC:CODES, descriptions, links)
    parsed from the `references_index.md` file.
    """
    references = doc_service.get_doc_references()
    if not references:
        # This could be an empty list if the file is empty or not found,
        # or we could raise an HTTPException if the file itself is critical and missing.
        # For now, returning an empty list is fine.
        pass 
    return references

@router.get("/content", response_model=Optional[Dict[str, str]])
async def get_documentation_page_content(
    path: str = Query(..., description="Relative path to the documentation markdown file from the docs root (e.g., 'preprocessing_pipeline.md')")
):
    """
    Retrieve the content of a specific documentation page, converted to HTML.
    The path should be the file path relative to the main 'docs' directory.
    Example: `preprocessing_pipeline.md` or `development_guides/debugging.md`
    """
    if not path or ".." in path: # Basic security check
        raise HTTPException(status_code=400, detail="Invalid path specified.")

    content = doc_service.get_doc_content(relative_path=path)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Documentation page not found at path: {path}")
    return content
