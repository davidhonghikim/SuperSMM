"""
Handles the conversion of sheet music images to MusicXML format using Audiveris.
"""

import logging
import os
import subprocess
import pathlib
from typing import Optional

from ..utils.logger import setup_logger

# Set up logger using the centralized configuration
logger = setup_logger(
    name=__name__,
    log_type="app",  # Log to the app log directory
    log_level=logging.DEBUG,  # Capture all levels in the log file
    log_to_console=True,  # Also log to console
    log_to_file=True  # Log to file
)

DEFAULT_AUDIVERIS_JAR_PATH = "src/lib/audiveris.jar"  # Use project-relative path


def convert_image_to_mxl_audiveris(
    image_path: str,
    output_directory: str,
    audiveris_jar_path: str = DEFAULT_AUDIVERIS_JAR_PATH,
) -> Optional[str]:
    """
    Converts a given image file to MusicXML format using Audiveris.

    Args:
        image_path (str): The path to the input image file (e.g., PNG, JPG).
        output_directory (str): The directory where the MusicXML (.mxl) file will be saved.
        audiveris_jar_path (str): The path to the Audiveris JAR file.
                                  Defaults to DEFAULT_AUDIVERIS_JAR_PATH.

    Returns:
        Optional[str]: The path to the generated MusicXML file if successful, otherwise None.
    """
    logger.info(f"Starting Audiveris conversion for image: {image_path}")
    logger.info(f"Using Audiveris JAR: {audiveris_jar_path}")
    logger.info(f"Output directory: {output_directory}")

    if not os.path.exists(audiveris_jar_path):
        logger.error(f"Audiveris JAR not found at: {audiveris_jar_path}")
        return None
    if not os.path.exists(image_path):
        logger.error(f"Input image not found at: {image_path}")
        return None
    if not os.path.isdir(output_directory):
        logger.error(
            f"Output directory does not exist or is not a directory: {output_directory}"
        )
        # Attempt to create the output directory if it doesn't exist
        try:
            logger.info(f"Attempting to create output directory: {output_directory}")
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            logger.error(
                f"Failed to create output directory {output_directory}: {e_mkdir}"
            )
            return None
        logger.info(f"Successfully created output directory: {output_directory}")

    base_name = pathlib.Path(image_path).stem

    # Construct classpath
    # audiveris_jar_path is absolute here due to prior processing in omr_processor.py
    jar_abs_path = pathlib.Path(audiveris_jar_path)
    src_dir = (
        jar_abs_path.parent
    )  # e.g., /Users/danger/CascadeProjects/LOO/SuperSMM/src
    project_root_dir = (
        src_dir.parent
    )  # e.g., /Users/danger/CascadeProjects/LOO/SuperSMM

    cp_entries = [str(jar_abs_path)]

    logger.info(f"Looking for lib dir. JAR path: {jar_abs_path}")
    logger.info(f"Calculated src_dir: {src_dir}")
    logger.info(f"Calculated project_root_dir: {project_root_dir}")

    # Check for lib directory in project root: SuperSMM/lib/
    project_lib_dir = project_root_dir / "lib"
    logger.info(
        f"Checking for project_lib_dir: {project_lib_dir} (Exists: {project_lib_dir.exists()}, IsDir: {project_lib_dir.is_dir()})"
    )
    if project_lib_dir.is_dir():
        cp_entries.append(str(project_lib_dir / "*"))
        logger.info(f"Adding to classpath: {project_lib_dir / '*'}")
    else:
        logger.info(
            f"Project lib dir not found or not a directory at {project_lib_dir}"
        )

    # Check for lib directory in src next to jar: SuperSMM/src/lib/
    src_lib_dir = src_dir / "lib"
    logger.info(
        f"Checking for src_lib_dir: {src_lib_dir} (Exists: {src_lib_dir.exists()}, IsDir: {src_lib_dir.is_dir()})"
    )
    if src_lib_dir.is_dir():
        cp_entries.append(str(src_lib_dir / "*"))
        logger.info(f"Adding to classpath: {src_lib_dir / '*'}")
    else:
        logger.info(f"Src lib dir not found or not a directory at {src_lib_dir}")
        if src_dir.exists() and src_dir.is_dir():
            try:
                logger.info(f"Contents of src_dir ({src_dir}): {os.listdir(src_dir)}")
            except Exception as e_ls:
                logger.error(f"Could not list contents of src_dir ({src_dir}): {e_ls}")
        else:
            logger.warning(f"src_dir ({src_dir}) does not exist or is not a directory.")

    classpath_str = os.pathsep.join(cp_entries)
    logger.info(f"Constructed classpath: {classpath_str}")

    main_class = "Audiveris"  # Deduced from stack trace

    command = [
        "java",
        "-cp",
        classpath_str,
        main_class,
        "-batch",
        # '-load', # Removed based on error: "-load" is not a valid option
        "-output",
        output_directory,
        "-export",
        image_path,  # Pass image_path as a positional argument at the end
    ]

    full_command_str = " ".join(command)
    logger.info(f"Executing Audiveris command: {full_command_str}")

    try:
        # Note: Using shell=False is safer. The classpath wildcard '*' is handled by Java, not the shell here.
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, encoding="utf-8"
        )

        # Print stdout and stderr to console for immediate feedback
        print("\n===== Audiveris STDOUT =====\n" + (process.stdout or "(empty)"))
        print("\n===== Audiveris STDERR =====\n" + (process.stderr or "(empty)"))

        if process.returncode != 0:
            logger.error(
                f"Audiveris process failed. See stdout/stderr above for details."
            )
            return None

        # After successful execution, locate the generated .mxl file.
        # Audiveris (>=5.3) using -output <dir> creates <dir>/<image_stem>/<image_stem>.mxl
        audiveris_output_subfolder = pathlib.Path(output_directory) / base_name
        expected_mxl_path = audiveris_output_subfolder / f"{base_name}.mxl"

        logger.info(f"Checking for expected MusicXML file at: {expected_mxl_path}")
        if expected_mxl_path.exists() and expected_mxl_path.is_file():
            logger.info(f"MusicXML file found: {expected_mxl_path}")
            return str(expected_mxl_path)
        else:
            # Fallback: check directly in output_directory
            fallback_mxl_path = pathlib.Path(output_directory) / f"{base_name}.mxl"
            if fallback_mxl_path.exists() and fallback_mxl_path.is_file():
                audiveris_output_subfolder.mkdir(parents=True, exist_ok=True)
                moved_path = audiveris_output_subfolder / f"{base_name}.mxl"
                fallback_mxl_path.rename(moved_path)
                logger.warning(
                    f"Moved MusicXML file to expected location: {moved_path}"
                )
                return str(moved_path)
            logger.error(
                f"MusicXML file NOT found at expected path: {expected_mxl_path}"
            )
            logger.error(
                f"Listing contents of Audiveris base output directory ({output_directory}):"
            )
            if (
                pathlib.Path(output_directory).exists()
                and pathlib.Path(output_directory).is_dir()
            ):
                for item in os.listdir(output_directory):
                    logger.error(
                        f"  - {item} (is_dir: {pathlib.Path(output_directory, item).is_dir()})"
                    )
            else:
                logger.error(
                    f"Audiveris base output directory does not exist or is not a directory."
                )

            logger.error(
                f"Listing contents of expected Audiveris output subfolder ({audiveris_output_subfolder}):"
            )
            if (
                audiveris_output_subfolder.exists()
                and audiveris_output_subfolder.is_dir()
            ):
                for item in os.listdir(audiveris_output_subfolder):
                    logger.error(
                        f"  - {item} (is_dir: {pathlib.Path(audiveris_output_subfolder, item).is_dir()})"
                    )
            else:
                logger.error(
                    f"Audiveris output subfolder does not exist or is not a directory."
                )
            return None

    except FileNotFoundError:
        logger.error(
            "Error: Java command not found. Please ensure Java JRE/JDK is installed and in your system PATH."
        )
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Audiveris conversion: {e}")
        logger.error(f"Command attempted: {full_command_str}")
        return None


if __name__ == "__main__":
    # Example Usage (for testing this module directly)
    # Create a dummy image file for testing
    print("Running Audiveris converter example...")
    test_image_path = "./dummy_sheet_music.png"
    test_output_dir = "./audiveris_output"

    # Create a dummy image file if it doesn't exist
    if not os.path.exists(test_image_path):
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (600, 800), color="white")
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Dummy Sheet Music for Audiveris Test", fill=(0, 0, 0))
            img.save(test_image_path)
            print(f"Created dummy image: {test_image_path}")
        except ImportError:
            print(
                "Pillow library not found, cannot create dummy image. Please create one manually for testing."
            )
            print(f"Please place a test image at: {os.path.abspath(test_image_path)}")
        except Exception as e:
            print(f"Could not create dummy image: {e}")

    # Ensure the dummy image exists before trying to convert
    if os.path.exists(test_image_path):
        # User needs to set their actual Audiveris JAR path for this test to work
        # For CI/automated tests, this path might need to be handled differently or Audiveris mocked.
        audiveris_jar = os.environ.get("AUDIVERIS_JAR_PATH", DEFAULT_AUDIVERIS_JAR_PATH)
        print(f"Using Audiveris JAR from: {audiveris_jar}")
        print(
            f"If this path is incorrect or Audiveris is not installed, the test will fail."
        )
        print(
            f"You can set the AUDIVERIS_JAR_PATH environment variable or modify DEFAULT_AUDIVERIS_JAR_PATH in the script."
        )

        mxl_file = convert_image_to_mxl_audiveris(
            os.path.abspath(test_image_path),
            os.path.abspath(test_output_dir),
            audiveris_jar,
        )

        if mxl_file:
            print(f"\nConversion successful! MusicXML saved to: {mxl_file}")
        else:
            print("\nConversion failed. Check logs for details.")
    else:
        print(f"Test image {test_image_path} not found. Skipping conversion test.")
