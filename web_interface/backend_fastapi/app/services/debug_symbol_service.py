# web_interface/backend_fastapi/app/services/debug_symbol_service.py
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional

# Path to the debug_symbols.py file
# Adjust this path if necessary based on how the FastAPI app is run.
DEBUG_SYMBOLS_FILE_PATH = Path("/Users/danger/CascadeProjects/LOO/SuperSMM/src/utils/debug_symbols.py")

class DebugSymbolService:
    def __init__(self):
        self.symbols_file_path = DEBUG_SYMBOLS_FILE_PATH

    def get_debug_symbols(self) -> List[Dict[str, Any]]:
        """
        Parses the debug_symbols.py file to extract DEBUG_MESSAGES.
        Returns a list of dictionaries, each representing a debug symbol.
        """
        symbols_list = []
        if not self.symbols_file_path.exists():
            # Log error or return empty, depending on desired behavior
            print(f"Error: Debug symbols file not found at {self.symbols_file_path}") # Replace with actual logging
            return symbols_list

        try:
            with open(self.symbols_file_path, 'r') as f:
                file_content = f.read()
            
            # Parse the Python file content into an AST
            tree = ast.parse(file_content)
            
            debug_messages_dict = None
            # Find the DEBUG_MESSAGES assignment
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'DEBUG_MESSAGES':
                            if isinstance(node.value, ast.Dict):
                                # Safely evaluate the dictionary literal
                                # This is safer than exec or eval on arbitrary code
                                debug_messages_dict = ast.literal_eval(node.value)
                                break
                    if debug_messages_dict is not None:
                        break
            
            if debug_messages_dict:
                for code, message_template in debug_messages_dict.items(): # Renamed 'details' to 'message_template'
                    # Ensure message_template is treated as a string
                    if isinstance(message_template, str):
                        symbols_list.append({
                            "code": code,
                            "message_template": message_template, # Use the string directly
                            "default_color": None # Color is not defined here, set to None
                        })
                    else:
                        # Log or handle unexpected format
                        print(f"Warning: Unexpected format for debug symbol '{code}'. Value is not a string: {message_template}")
            else:
                print("Error: DEBUG_MESSAGES dictionary not found or not in expected format.") # Replace with logging

        except FileNotFoundError:
            print(f"Error: Debug symbols file not found at {self.symbols_file_path}") # Replace with logging
        except SyntaxError as e:
            print(f"Error parsing debug_symbols.py (SyntaxError): {e}") # Replace with logging
        except Exception as e:
            print(f"An unexpected error occurred while processing debug symbols: {e}") # Replace with logging
            
        return symbols_list

# Example usage (for testing the service directly)
if __name__ == "__main__":
    service = DebugSymbolService()
    print("--- Debug Symbols ---")
    symbols = service.get_debug_symbols()
    if symbols:
        for s in symbols:
            print(s)
    else:
        print("No symbols found or error in processing.")
