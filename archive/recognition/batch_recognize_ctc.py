import csv
import subprocess
from pathlib import Path
import os


def main():
    project_root = Path(__file__).parent
    ctc_predict_script = project_root / "ml/models/resources/tf-deep-omr/ctc_predict.py"
    agnostic_model_path = (
        project_root
        / "ml/models/resources/tf-deep-omr/Data/Models/Agnostic Model/agnostic_model.meta"
    )
    agnostic_vocab_path = (
        project_root / "ml/models/resources/tf-deep-omr/Data/vocabulary_agnostic.txt"
    )

    # Assuming candidates are processed for page_1 for now
    # TODO: Make page_num a parameter or process all pages
    candidates_dir = project_root / "data/output/symbols/page_1/candidates"
    output_csv_path = (
        project_root / "data/output/symbols/page_1/recognized_candidates_ctc.csv"
    )

    if not candidates_dir.exists():
        print(f"[ERROR] Candidates directory not found: {candidates_dir}")
        return

    candidate_images = sorted(list(candidates_dir.glob("*.png")))

    if not candidate_images:
        print(f"[INFO] No candidate images found in {candidates_dir}")
        return

    print(f"[INFO] Found {len(candidate_images)} candidate images in {candidates_dir}")
    print(f"[INFO] Using model: {agnostic_model_path}")
    print(f"[INFO] Using vocabulary: {agnostic_vocab_path}")
    print(f"[INFO] Output will be saved to: {output_csv_path}")

    with open(output_csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["candidate_image_filename", "recognized_symbols"])

        for i, image_path in enumerate(candidate_images):
            command = [
                "python",
                str(ctc_predict_script),
                "-image",
                str(image_path),
                "-model",
                str(agnostic_model_path),
                "-vocabulary",
                str(agnostic_vocab_path),
            ]

            print(
                f"\n[INFO] Processing image {i+1}/{len(candidate_images)}: {image_path.name}"
            )
            # print(f"Running command: {' '.join(command)}") # For debugging

            try:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=project_root,
                )
                output_lines = process.stdout.splitlines()
                recognized_text = ""

                # Find the line with recognized symbols
                for line_idx, line in enumerate(output_lines):
                    if "Recognized symbols:" in line:
                        # The next line should contain the tab-separated symbols
                        if line_idx + 1 < len(output_lines):
                            recognized_text = output_lines[line_idx + 1].strip()
                            # Replace tabs with spaces for easier CSV reading, or keep as is
                            # recognized_text = recognized_text.replace('\t', ' ')
                        break

                if recognized_text:
                    print(f"[RESULT] Recognized: {recognized_text}")
                    csv_writer.writerow([image_path.name, recognized_text])
                else:
                    print(
                        "[WARN] 'Recognized symbols:' line not found or no symbols followed."
                    )
                    csv_writer.writerow([image_path.name, "ERROR_NO_RECOGNITION_LINE"])

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to process {image_path.name}")
                print(f"Stderr: {e.stderr}")
                csv_writer.writerow(
                    [image_path.name, f"ERROR_SUBPROCESS: {e.returncode}"]
                )
            except Exception as e:
                print(
                    f"[ERROR] An unexpected error occurred with {image_path.name}: {e}"
                )
                csv_writer.writerow([image_path.name, "ERROR_UNEXPECTED"])

    print(f"\n[INFO] Batch recognition complete. Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
