#!/bin/bash

# Process Somewhere Over the Rainbow sheet music using the OMR CLI tool

# Set up directories
OUTPUT_DIR="output/rainbow_cli"
mkdir -p "$OUTPUT_DIR"

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Processing Somewhere Over the Rainbow with OMR CLI ===${NC}"

# Process page 1 with Agnostic model
echo -e "${BLUE}Processing page 1 with Agnostic model...${NC}"
python omr_cli.py --test_image Data/Somewhere_Over_the_Rainbow_page_1.png --model agnostic --output "$OUTPUT_DIR/agnostic_page1"

# Process page 1 with Semantic model
echo -e "${BLUE}Processing page 1 with Semantic model...${NC}"
python omr_cli.py --test_image Data/Somewhere_Over_the_Rainbow_page_1.png --model semantic --output "$OUTPUT_DIR/semantic_page1"

# Process page 2 with Agnostic model
echo -e "${BLUE}Processing page 2 with Agnostic model...${NC}"
python omr_cli.py --test_image Data/Somewhere_Over_the_Rainbow_page_2.png --model agnostic --output "$OUTPUT_DIR/agnostic_page2"

# Process page 2 with Semantic model
echo -e "${BLUE}Processing page 2 with Semantic model...${NC}"
python omr_cli.py --test_image Data/Somewhere_Over_the_Rainbow_page_2.png --model semantic --output "$OUTPUT_DIR/semantic_page2"

# Batch process all pages
echo -e "${BLUE}Batch processing all pages...${NC}"
python omr_cli.py --batch_test --image_dir Data --image_pattern "Somewhere_Over_the_Rainbow_page_*.png" --output "$OUTPUT_DIR/batch_results"

# Generate HTML report
echo -e "${YELLOW}Generating HTML report...${NC}"

cat > "$OUTPUT_DIR/report.html" << HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Somewhere Over the Rainbow - OMR CLI Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-container {
            margin-top: 20px;
        }
        .image-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .model-results {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .model-card {
            flex: 1;
            min-width: 300px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
        }
        .model-header {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }
        .footer {
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }
        pre {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 14px;
        }
        .btn {
            display: inline-block;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
        }
        .btn:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Somewhere Over the Rainbow - OMR CLI Results</h1>
        <p>Generated on: $(date '+%Y-%m-%d %H:%M:%S')</p>
        
        <div class="card">
            <h2>Original Sheet Music</h2>
            <div class="model-results">
                <div class="model-card">
                    <div class="model-header">Page 1</div>
                    <div class="image-container">
                        <img src="../Data/Somewhere_Over_the_Rainbow_page_1.png" alt="Page 1">
                    </div>
                </div>
                <div class="model-card">
                    <div class="model-header">Page 2</div>
                    <div class="image-container">
                        <img src="../Data/Somewhere_Over_the_Rainbow_page_2.png" alt="Page 2">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Page 1 Results</h2>
            <div class="model-results">
                <div class="model-card">
                    <div class="model-header">Agnostic Model</div>
                    <div class="image-container">
                        <img src="agnostic_page1.png" alt="Agnostic Model - Page 1">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "$OUTPUT_DIR/agnostic_page1.txt" 2>/dev/null || echo "No results available")</pre>
                    <a href="agnostic_page1.json" class="btn">View JSON</a>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <div class="image-container">
                        <img src="semantic_page1.png" alt="Semantic Model - Page 1">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "$OUTPUT_DIR/semantic_page1.txt" 2>/dev/null || echo "No results available")</pre>
                    <a href="semantic_page1.json" class="btn">View JSON</a>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Page 2 Results</h2>
            <div class="model-results">
                <div class="model-card">
                    <div class="model-header">Agnostic Model</div>
                    <div class="image-container">
                        <img src="agnostic_page2.png" alt="Agnostic Model - Page 2">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "$OUTPUT_DIR/agnostic_page2.txt" 2>/dev/null || echo "No results available")</pre>
                    <a href="agnostic_page2.json" class="btn">View JSON</a>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <div class="image-container">
                        <img src="semantic_page2.png" alt="Semantic Model - Page 2">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "$OUTPUT_DIR/semantic_page2.txt" 2>/dev/null || echo "No results available")</pre>
                    <a href="semantic_page2.json" class="btn">View JSON</a>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Batch Processing Results</h2>
            <p>The batch processing feature of the OMR CLI tool processes multiple images and generates a comprehensive report.</p>
            <h3>Batch Results:</h3>
            <pre>$(cat "$OUTPUT_DIR/batch_results/batch_report.txt" 2>/dev/null || echo "No batch results available")</pre>
            <a href="batch_results/batch_report.html" class="btn">View Batch HTML Report</a>
            <a href="batch_results/batch_report.json" class="btn">View Batch JSON Report</a>
        </div>
        
        <div class="card">
            <h2>Model Comparison</h2>
            <p>The Agnostic model recognizes individual symbols with their positions, while the Semantic model provides a higher-level musical interpretation with pitch information.</p>
            
            <h3>Agnostic Model</h3>
            <p><strong>Pros:</strong> More detailed about exact positioning of each element</p>
            <p><strong>Cons:</strong> Less musically meaningful without pitch information</p>
            <p><strong>Example output:</strong> clef.C-L1, accidental.flat-L4, note.eighth-L4</p>
            
            <h3>Semantic Model</h3>
            <p><strong>Pros:</strong> More musically meaningful with pitch information</p>
            <p><strong>Cons:</strong> Less detailed about exact symbol positioning</p>
            <p><strong>Example output:</strong> clef-C1, keySignature-EbM, note-Bb4_eighth</p>
        </div>
        
        <div class="card">
            <h2>Next Steps</h2>
            <p>To improve the recognition results for "Somewhere Over the Rainbow", consider the following:</p>
            <ol>
                <li>Try different image preprocessing techniques (binarization, deskewing, etc.)</li>
                <li>Adjust the image resolution to match the training data</li>
                <li>Fine-tune the models with similar sheet music examples</li>
                <li>Use the web visualizer for interactive exploration of the results</li>
            </ol>
            <a href="../omr_visualizer.py" class="btn">Launch Web Visualizer</a>
        </div>
        
        <div class="footer">
            <p>Generated by OMR CLI Processing System</p>
        </div>
    </div>
</body>
</html>
HTML

echo -e "${GREEN}=== Processing completed successfully! ===${NC}"
echo -e "Results saved to $OUTPUT_DIR/"
echo -e "HTML report: $OUTPUT_DIR/report.html"

# Launch the web visualizer
echo -e "${YELLOW}Launching web visualizer for interactive exploration...${NC}"
echo -e "Run the following command to start the web visualizer:"
echo -e "${GREEN}python omr_visualizer.py${NC}"
