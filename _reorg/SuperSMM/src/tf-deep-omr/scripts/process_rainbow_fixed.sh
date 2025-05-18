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
python omr_cli.py test -model agnostic -image Data/Somewhere_Over_the_Rainbow_page_1.png

# Process page 1 with Semantic model
echo -e "${BLUE}Processing page 1 with Semantic model...${NC}"
python omr_cli.py test -model semantic -image Data/Somewhere_Over_the_Rainbow_page_1.png

# Process page 2 with Agnostic model
echo -e "${BLUE}Processing page 2 with Agnostic model...${NC}"
python omr_cli.py test -model agnostic -image Data/Somewhere_Over_the_Rainbow_page_2.png

# Process page 2 with Semantic model
echo -e "${BLUE}Processing page 2 with Semantic model...${NC}"
python omr_cli.py test -model semantic -image Data/Somewhere_Over_the_Rainbow_page_2.png

# Batch process all pages
echo -e "${BLUE}Batch processing all pages...${NC}"
python omr_cli.py batch -all-models -directory Data -output "$OUTPUT_DIR/batch_report.json"

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
                        <img src="../Data/Somewhere_Over_the_Rainbow_page_1_agnostic.png" alt="Agnostic Model - Page 1" onerror="this.src='../static/placeholder.png'">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "../Data/Somewhere_Over_the_Rainbow_page_1_agnostic.txt" 2>/dev/null || echo "No results available")</pre>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <div class="image-container">
                        <img src="../Data/Somewhere_Over_the_Rainbow_page_1_semantic.png" alt="Semantic Model - Page 1" onerror="this.src='../static/placeholder.png'">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "../Data/Somewhere_Over_the_Rainbow_page_1_semantic.txt" 2>/dev/null || echo "No results available")</pre>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Page 2 Results</h2>
            <div class="model-results">
                <div class="model-card">
                    <div class="model-header">Agnostic Model</div>
                    <div class="image-container">
                        <img src="../Data/Somewhere_Over_the_Rainbow_page_2_agnostic.png" alt="Agnostic Model - Page 2" onerror="this.src='../static/placeholder.png'">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "../Data/Somewhere_Over_the_Rainbow_page_2_agnostic.txt" 2>/dev/null || echo "No results available")</pre>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <div class="image-container">
                        <img src="../Data/Somewhere_Over_the_Rainbow_page_2_semantic.png" alt="Semantic Model - Page 2" onerror="this.src='../static/placeholder.png'">
                    </div>
                    <h4>Recognition Results:</h4>
                    <pre>$(cat "../Data/Somewhere_Over_the_Rainbow_page_2_semantic.txt" 2>/dev/null || echo "No results available")</pre>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Batch Processing Results</h2>
            <p>The batch processing feature of the OMR CLI tool processes multiple images and generates a comprehensive report.</p>
            <a href="batch_report.json" class="btn">View Batch JSON Report</a>
        </div>
        
        <div class="card">
            <h2>Interactive Visualization</h2>
            <p>To explore the results interactively, launch the web visualizer:</p>
            <code>python omr_cli.py visualize</code>
            <p>This will start a web server that allows you to upload images and visualize the recognition results.</p>
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
echo -e "${GREEN}python omr_cli.py visualize${NC}"
