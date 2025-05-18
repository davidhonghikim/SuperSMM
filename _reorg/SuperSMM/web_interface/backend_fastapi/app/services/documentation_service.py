# web_interface/backend_fastapi/app/services/documentation_service.py
import markdown
import re
from pathlib import Path
from typing import List, Dict, Optional

# Assuming the script runs from a context where SuperSMM is the root or accessible
# Adjust this path if necessary based on how the FastAPI app is run.
DOCS_ROOT_PATH = Path("/Users/danger/CascadeProjects/LOO/SuperSMM/docs")
REFERENCES_INDEX_FILE = DOCS_ROOT_PATH / "references_index.md"

class DocumentationService:
    def __init__(self):
        self.references_index_path = REFERENCES_INDEX_FILE
        self.docs_root_path = DOCS_ROOT_PATH

    def get_doc_references(self) -> List[Dict[str, str]]:
        """
        Parses the references_index.md file to extract DOC:CODE, link, and description.
        Returns a list of dictionaries, each representing a documentation reference.
        """
        references = []
        if not self.references_index_path.exists():
            return references # Or raise an error

        # Regex to capture: *   **`DOC:CODE`**: [description text](link) - further description
        # It captures the DOC:CODE, the link, and the combined description text.
        # Simpler regex: `\*\s*\*\*`DOC:([^`]+)`\*\*`:\s*\[([^\]]+)\]\(([^\)]+)\)(.*)`
        # Breakdown:
        # \*\s*\*\*(`DOC:[^`]+`)\*\*  : Captures the DOC code like `DOC:PREP-STAFF-001`
        # :\s*\[([^\]]+)\]             : Captures the link text (description for the link itself)
        # \(([^\)]+)\)                  : Captures the actual link (URL/path)
        # (.*)                          : Captures the rest of the line as further description
        pattern = re.compile(r"^\*\s*\*\*(`DOC:[^`]+`)\*\*:\s*\[([^\]]+)\]\(([^\)]+)\)(.*)$")

        with open(self.references_index_path, 'r') as f:
            for line in f:
                line = line.strip()
                match = pattern.match(line)
                if match:
                    doc_code = match.group(1).strip("`")
                    link_text = match.group(2).strip()
                    link_href = match.group(3).strip()
                    additional_desc = match.group(4).strip(" - ").strip()
                    
                    full_description = link_text
                    if additional_desc:
                        full_description += f" - {additional_desc}"

                    references.append({
                        "code": doc_code,
                        "description": full_description,
                        "link": link_href # This link is relative to the docs folder or absolute
                    })
        return references

    def get_doc_content(self, relative_path: str) -> Optional[Dict[str, str]]:
        """
        Reads the content of a markdown file specified by its relative path from the docs root.
        Returns a dictionary with the raw markdown and HTML content, or None if not found.
        """
        # Security: Ensure the relative_path doesn't try to escape the DOCS_ROOT_PATH
        # Path.resolve() can help, but careful concatenation is key.
        # For now, assume relative_path is trusted or validated by the router.
        doc_file_path = (self.docs_root_path / relative_path).resolve()

        # Basic check to prevent path traversal above docs_root_path
        if self.docs_root_path not in doc_file_path.parents and doc_file_path != self.docs_root_path:
             # This check might need refinement depending on symlinks etc.
             # A more robust way is to ensure doc_file_path.is_relative_to(self.docs_root_path) (Python 3.9+)
             # For now, let's assume direct children or files in subdirs are fine.
             pass # Allow if it's within the tree, will be caught by exists() if invalid

        if doc_file_path.exists() and doc_file_path.is_file():
            try:
                raw_markdown = doc_file_path.read_text()
                html_content = markdown.markdown(raw_markdown, extensions=['fenced_code', 'tables', 'toc'])
                return {
                    "path": str(relative_path),
                    "markdown": raw_markdown,
                    "html": html_content
                }
            except Exception as e:
                # Log this error
                print(f"Error reading or converting doc {relative_path}: {e}") # Replace with actual logging
                return None
        return None

# Example usage (for testing the service directly)
if __name__ == "__main__":
    service = DocumentationService()
    print("--- References ---")
    refs = service.get_doc_references()
    for r in refs:
        print(r)
    
    print("\n--- Content of first link (if any) ---")
    if refs:
        # Taking the link from the first reference to test get_doc_content
        # This link might be to an anchor, so we split it off
        first_link_path = refs[0]['link'].split('#')[0]
        if first_link_path:
            content = service.get_doc_content(first_link_path)
            if content:
                print(f"Content for: {content['path']}")
                # print(content['html']) # HTML can be very long
            else:
                print(f"Could not retrieve content for: {first_link_path}")
        else:
            print("First reference link is empty or only an anchor.")
