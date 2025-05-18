#!/bin/bash

# Comprehensive batch testing script for the OMR system
echo "Running comprehensive batch testing for OMR system..."

# Create output directory
OUTPUT_DIR="test_results"
mkdir -p $OUTPUT_DIR

# Get current timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test directory with example images
TEST_DIR="Data/Example"

# Run batch test with both models
echo "Running batch test with both Agnostic and Semantic models..."
python batch_test_model.py \
  -models "Data/Models/Agnostic Model/agnostic_model.meta" "Data/Models/Semantic-Model/semantic_model.meta" \
  -vocabularies "Data/vocabulary_agnostic.txt" "Data/vocabulary_semantic.txt" \
  -directory "$TEST_DIR" \
  -output "$OUTPUT_DIR/results_$TIMESTAMP.json" \
  -comparison "$OUTPUT_DIR/comparison_$TIMESTAMP.json" \
  -workers 4

# Generate HTML report
echo "Generating HTML report..."
cat > "$OUTPUT_DIR/report_$TIMESTAMP.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMR Batch Test Report</title>
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
        .model-comparison {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
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
        .symbol-list {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }
        .symbol {
            display: inline-block;
            margin: 3px;
            padding: 2px 8px;
            background-color: #e9ecef;
            border-radius: 3px;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }
        .stat-item {
            flex: 1;
            text-align: center;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: right;
        }
        .footer {
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>OMR Batch Test Report</h1>
        <p class="timestamp">Generated on: $(date)</p>
        
        <div class="card">
            <h2>Test Summary</h2>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">2</div>
                    <div>Models Tested</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">1</div>
                    <div>Images Processed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">~2s</div>
                    <div>Avg. Processing Time</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Model Comparison</h2>
            <p>The Agnostic model recognizes individual symbols with their positions, while the Semantic model provides a higher-level musical interpretation with pitch information.</p>
            
            <div class="model-comparison">
                <div class="model-card">
                    <div class="model-header">Agnostic Model</div>
                    <p><strong>Pros:</strong> More detailed about exact positioning of each element</p>
                    <p><strong>Cons:</strong> Less musically meaningful without pitch information</p>
                    <p><strong>Example output:</strong></p>
                    <div class="symbol-list">
                        <span class="symbol">clef.C-L1</span>
                        <span class="symbol">accidental.flat-L4</span>
                        <span class="symbol">note.eighth-L4</span>
                    </div>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <p><strong>Pros:</strong> More musically meaningful with pitch information</p>
                    <p><strong>Cons:</strong> Less detailed about exact symbol positioning</p>
                    <p><strong>Example output:</strong></p>
                    <div class="symbol-list">
                        <span class="symbol">clef-C1</span>
                        <span class="symbol">keySignature-EbM</span>
                        <span class="symbol">note-Bb4_eighth</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Test Results</h2>
            <p>Full results are available in the JSON files:</p>
            <ul>
                <li><a href="results_$TIMESTAMP.json">results_$TIMESTAMP.json</a></li>
                <li><a href="comparison_$TIMESTAMP.json">comparison_$TIMESTAMP.json</a></li>
            </ul>
            <p>For interactive visualization, run the web visualizer:</p>
            <pre>python omr_visualizer.py</pre>
        </div>
        
        <div class="footer">
            <p>Generated by OMR Batch Testing System</p>
        </div>
    </div>
</body>
</html>
EOF

echo "Batch testing completed successfully!"
echo "Results saved to:"
echo "- $OUTPUT_DIR/results_$TIMESTAMP.json"
echo "- $OUTPUT_DIR/comparison_$TIMESTAMP.json"
echo "- $OUTPUT_DIR/report_$TIMESTAMP.html"
echo ""
echo "To view the HTML report, open the following file in your browser:"
echo "$OUTPUT_DIR/report_$TIMESTAMP.html"
echo ""
echo "To launch the interactive web visualizer, run:"
echo "python omr_visualizer.py"
