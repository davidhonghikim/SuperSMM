import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from utils.logging_utils import get_logger
logger = get_logger('lint_and_fix', 'linting', structured=True)
logger.info(sys.stdin.read())
