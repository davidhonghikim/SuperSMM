# src/utils/debug_symbols.py

DEBUG_MESSAGES = {
    # General Errors
    "ERR_INPUT_NONE": "Error: Critical input was None.",
    "ERR_FILE_NOT_FOUND": "Error: Specified file not found: {filepath}",
    "ERR_CONFIG_MISSING": "Error: Configuration key missing: {key}",
    # Preprocessing Stage
    "PREPROC_IMG_LOAD_FAIL": "Preprocessing: Failed to load image: {image_path}",
    "PREPROC_NORM_START": "Preprocessing: Image normalization started.",
    "PREPROC_NORM_END": "Preprocessing: Image normalization completed.",
    "PREPROC_BIN_START": "Preprocessing: Binarization started with method: {method}.",
    "PREPROC_BIN_END": "Preprocessing: Binarization completed.",
    "PREPROC_STAFF_DETECT_START": "Preprocessing: Staff line detection initiated.",
    "PREPROC_STAFF_DETECT_END": "Preprocessing: Staff line detection finished. Lines found: {count}",
    # Segmentation Stage
    "SEG_START": "Segmentation: Process started for page: {page_id}",
    "SEG_NO_STAFF": "Segmentation: No staff lines found or provided to segment.",
    "SEG_SYMBOLS_START": "Segmentation: Symbol segmentation starting.",
    "SEG_SYMBOLS_END": "Segmentation: Symbol segmentation finished. Symbols found: {count}",
    # Generic Debugging Points
    "DBG_POINT_A": "Debug: Reached point A in {function_name}.",
    "DBG_POINT_B": "Debug: Reached point B in {function_name}. Value: {value}",
    "DBG_VAR_STATE": "Debug: Variable {var_name} in {function_name} is {var_value}",
}


# ANSI escape codes for colors (optional, for console visibility)
class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    ENDC = "\033[0m"


def get_message(symbol_code: str, color: str = None, **kwargs) -> str:
    """
    Retrieves a formatted debug message for a given symbol code.

    Args:
        symbol_code (str): The short code for the debug message.
        color (str, optional): ANSI color code from Colors class. Defaults to None.
        **kwargs: Values to format into the message string.

    Returns:
        str: The formatted debug message, or an error if the symbol is not found.
    """
    message_template = DEBUG_MESSAGES.get(symbol_code)
    if message_template is None:
        return f"{Colors.RED}Unknown debug symbol: {symbol_code}{Colors.ENDC}"

    try:
        formatted_message = message_template.format(**kwargs)
    except KeyError as e:
        return f"{Colors.RED}Missing key for symbol {symbol_code}: {e}. Args provided: {kwargs}{Colors.ENDC}"

    if color:
        return f"{color}{formatted_message}{Colors.ENDC}"
    return formatted_message


# --- Example Usage (can be removed or kept for testing) ---
if __name__ == "__main__":
    print(get_message("ERR_INPUT_NONE", color=Colors.RED))
    print(get_message("PREPROC_BIN_START", method="Otsu", color=Colors.BLUE))
    print(
        get_message(
            "DBG_VAR_STATE",
            var_name="my_var",
            function_name="test_func",
            var_value=42,
            color=Colors.GREEN,
        )
    )
    print(get_message("UNKNOWN_SYMBOL"))  # Test unknown symbol
    print(
        get_message("ERR_FILE_NOT_FOUND", color=Colors.YELLOW)
    )  # Test missing format args
