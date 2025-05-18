import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from core.omr_pipeline import OMRPipeline
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    pipeline = OMRPipeline()
    try:
        result = pipeline.process_sheet_music(
            "/tmp/this_file_does_not_exist.png")
        print("Returned result:", result)
    except Exception as e:
        print("Exception raised:", e)
    print("Check /tmp/omr_debug.txt for method entry log.")
