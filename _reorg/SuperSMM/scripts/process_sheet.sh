#!/bin/bash
# Script to process a sheet music PDF through the entire OMR pipeline
# without using Audiveris, using our trained model for symbol recognition

# Exit on error
set -e

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the root directory of the project
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
INPUT_PDF=""
OUTPUT_DIR="${ROOT_DIR}/output"
DEBUG_DIR="${ROOT_DIR}/debug_output"
MODEL_DIR="${ROOT_DIR}/src/tf-deep-omr/model/primus_model"
CONFIG_DIR="${ROOT_DIR}/config"
TEMP_DIR="${ROOT_DIR}/output/temp"

# Parse command line arguments
function usage {
    echo "Usage: $0 -i input_pdf [-o output_dir] [-d debug_dir] [-m model_dir] [-c config_dir]"
    echo "  -i input_pdf    : Path to the input PDF file"
    echo "  -o output_dir   : Path to the output directory (default: ${OUTPUT_DIR})"
    echo "  -d debug_dir    : Path to the debug output directory (default: ${DEBUG_DIR})"
    echo "  -m model_dir    : Path to the model directory (default: ${MODEL_DIR})"
    echo "  -c config_dir   : Path to the config directory (default: ${CONFIG_DIR})"
    echo "  -h              : Show this help message"
    exit 1
}

while getopts "i:o:d:m:c:h" opt; do
    case ${opt} in
        i )
            INPUT_PDF="$OPTARG"
            ;;
        o )
            OUTPUT_DIR="$OPTARG"
            ;;
        d )
            DEBUG_DIR="$OPTARG"
            ;;
        m )
            MODEL_DIR="$OPTARG"
            ;;
        c )
            CONFIG_DIR="$OPTARG"
            ;;
        h )
            usage
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if input PDF is provided
if [ -z "$INPUT_PDF" ]; then
    echo "Error: Input PDF file is required"
    usage
fi

# Check if input PDF exists
if [ ! -f "$INPUT_PDF" ]; then
    echo "Error: Input PDF file '$INPUT_PDF' does not exist"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}/sheets"
mkdir -p "${OUTPUT_DIR}/symbols"
mkdir -p "${OUTPUT_DIR}/musicxml"
mkdir -p "${DEBUG_DIR}/segmentation"
mkdir -p "${TEMP_DIR}"

# Get the basename of the input PDF
PDF_BASENAME=$(basename "${INPUT_PDF%.pdf}")
echo "Processing PDF: ${PDF_BASENAME}"

# Step 1: Convert PDF to images
echo "Step 1: Converting PDF to images..."
python "${ROOT_DIR}/src/preprocessing/pdf_to_images.py" \
    --input "${INPUT_PDF}" \
    --output "${OUTPUT_DIR}/sheets/${PDF_BASENAME}" \
    --dpi 300

# Step 2: Preprocess images
echo "Step 2: Preprocessing images..."
for img in "${OUTPUT_DIR}/sheets/${PDF_BASENAME}"/*.png; do
    page_name=$(basename "${img%.png}")
    echo "  Processing page: ${page_name}"
    
    python "${ROOT_DIR}/src/preprocessing/preprocess_image.py" \
        --input "${img}" \
        --output "${OUTPUT_DIR}/sheets/${PDF_BASENAME}/${page_name}_processed.png" \
        --debug-dir "${DEBUG_DIR}/segmentation/${PDF_BASENAME}" \
        --config "${CONFIG_DIR}/preprocessing.yaml"
done

# Step 3: Segment staves
echo "Step 3: Segmenting staves..."
for img in "${OUTPUT_DIR}/sheets/${PDF_BASENAME}"/*_processed.png; do
    page_name=$(basename "${img%_processed.png}")
    echo "  Segmenting staves in page: ${page_name}"
    
    python "${ROOT_DIR}/src/segmentation/segment_staves.py" \
        --input "${img}" \
        --output "${OUTPUT_DIR}/symbols/${PDF_BASENAME}/${page_name}" \
        --debug-dir "${DEBUG_DIR}/segmentation/${PDF_BASENAME}/${page_name}" \
        --config "${CONFIG_DIR}/segmentation.yaml"
done

# Step 4: Recognize symbols
echo "Step 4: Recognizing symbols..."
for staff_dir in "${OUTPUT_DIR}/symbols/${PDF_BASENAME}"/*; do
    if [ -d "${staff_dir}" ]; then
        staff_name=$(basename "${staff_dir}")
        echo "  Recognizing symbols in staff: ${staff_name}"
        
        # Create output directory for recognized symbols
        mkdir -p "${OUTPUT_DIR}/symbols/${PDF_BASENAME}/${staff_name}/recognized"
        
        python "${ROOT_DIR}/src/recognition/recognize_symbols.py" \
            --input "${staff_dir}" \
            --output "${OUTPUT_DIR}/symbols/${PDF_BASENAME}/${staff_name}/recognized" \
            --model "${MODEL_DIR}/model" \
            --vocabulary "${ROOT_DIR}/src/tf-deep-omr/data/vocabulary_semantic.txt" \
            --config "${CONFIG_DIR}/recognition.yaml"
    fi
done

# Step 5: Generate MusicXML
echo "Step 5: Generating MusicXML..."
for page_dir in "${OUTPUT_DIR}/symbols/${PDF_BASENAME}"; do
    page_name=$(basename "${page_dir}")
    echo "  Generating MusicXML for page: ${page_name}"
    
    python "${ROOT_DIR}/src/export/generate_musicxml.py" \
        --input "${OUTPUT_DIR}/symbols/${PDF_BASENAME}/${page_name}" \
        --output "${OUTPUT_DIR}/musicxml/${PDF_BASENAME}_${page_name}.xml" \
        --config "${CONFIG_DIR}/export.yaml"
done

# Cleanup temporary files if needed
echo "Cleaning up temporary files..."
rm -rf "${TEMP_DIR}"

echo "Processing complete!"
echo "Output MusicXML files are in ${OUTPUT_DIR}/musicxml/"
echo "Debug outputs are in ${DEBUG_DIR}/segmentation/${PDF_BASENAME}/"

# Print a summary of the results
echo ""
echo "Summary:"
echo "--------"
echo "Input PDF: ${INPUT_PDF}"
echo "Output MusicXML: ${OUTPUT_DIR}/musicxml/${PDF_BASENAME}*.xml"
echo "Number of pages processed: $(ls -1 "${OUTPUT_DIR}/sheets/${PDF_BASENAME}"/*_processed.png | wc -l)"
echo "Number of MusicXML files generated: $(ls -1 "${OUTPUT_DIR}/musicxml/${PDF_BASENAME}"*.xml | wc -l 2>/dev/null || echo 0)"
echo ""
echo "To view the MusicXML files, you can use any MusicXML viewer like MuseScore or Finale."

# Exit with success
exit 0 