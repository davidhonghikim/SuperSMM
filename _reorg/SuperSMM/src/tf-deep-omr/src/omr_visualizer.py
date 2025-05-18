#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web-based visualizer for OMR results using Flask
"""

import argparse
import os
import json
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess
import threading
import webbrowser
import time
from batch_test_model import OMRModelTester

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables
MODELS = {}
RESULTS_CACHE = {}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', models=list(MODELS.keys()))

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/process', methods=['POST'])
def process_image():
    """Process an image with the selected model"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    model_name = request.form.get('model', list(MODELS.keys())[0])
    
    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 400
    
    # Save the uploaded file temporarily
    temp_path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    
    # Check if we have cached results
    cache_key = f"{temp_path}_{model_name}"
    if cache_key in RESULTS_CACHE:
        return jsonify(RESULTS_CACHE[cache_key])
    
    # Process the image
    model_tester = MODELS[model_name]
    result = model_tester.process_image(temp_path)
    
    # Add image data for display
    img = cv2.imread(temp_path)
    height, width = img.shape[:2]
    
    # Create a visualization of the symbols on the image
    vis_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)  # Red
    thickness = 1
    
    # Add symbols as text overlay
    y_pos = 20
    for i, symbol in enumerate(result['symbols']):
        # Position text in a grid-like pattern
        x_pos = 10 + (i % 5) * (width // 5)
        if i > 0 and i % 5 == 0:
            y_pos += 20
        
        cv2.putText(vis_img, symbol, (x_pos, y_pos), 
                    font, font_scale, color, thickness)
    
    # Save visualization
    vis_path = os.path.join('static', 'results', f"vis_{file.filename}")
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    cv2.imwrite(vis_path, vis_img)
    
    # Convert images to base64 for embedding in JSON
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    _, vis_buffer = cv2.imencode('.png', vis_img)
    vis_base64 = base64.b64encode(vis_buffer).decode('utf-8')
    
    # Prepare response
    response = {
        'original_image': img_base64,
        'visualization': vis_base64,
        'symbols': result['symbols'],
        'processing_time': result['processing_time'],
        'model': model_name,
        'image_path': temp_path,
        'vis_path': vis_path
    }
    
    # Cache the results
    RESULTS_CACHE[cache_key] = response
    
    return jsonify(response)

@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare results from all models on the same image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Save the uploaded file temporarily
    temp_path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    
    # Process with all models
    results = []
    for model_name, model_tester in MODELS.items():
        # Check if we have cached results
        cache_key = f"{temp_path}_{model_name}"
        if cache_key in RESULTS_CACHE:
            results.append(RESULTS_CACHE[cache_key])
            continue
        
        # Process the image
        result = model_tester.process_image(temp_path)
        
        # Add image data for display
        img = cv2.imread(temp_path)
        
        # Create a visualization of the symbols on the image
        vis_img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 255)  # Red
        thickness = 1
        
        # Add symbols as text overlay
        y_pos = 20
        for i, symbol in enumerate(result['symbols']):
            # Position text in a grid-like pattern
            x_pos = 10 + (i % 5) * (img.shape[1] // 5)
            if i > 0 and i % 5 == 0:
                y_pos += 20
            
            cv2.putText(vis_img, symbol, (x_pos, y_pos), 
                        font, font_scale, color, thickness)
        
        # Save visualization
        vis_path = os.path.join('static', 'results', f"{model_name}_vis_{file.filename}")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, vis_img)
        
        # Convert images to base64 for embedding in JSON
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        _, vis_buffer = cv2.imencode('.png', vis_img)
        vis_base64 = base64.b64encode(vis_buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'original_image': img_base64,
            'visualization': vis_base64,
            'symbols': result['symbols'],
            'processing_time': result['processing_time'],
            'model': model_name,
            'image_path': temp_path,
            'vis_path': vis_path
        }
        
        # Cache the results
        RESULTS_CACHE[cache_key] = response
        results.append(response)
    
    # Calculate comparison metrics
    comparison = {
        'models': list(MODELS.keys()),
        'symbol_counts': {r['model']: len(r['symbols']) for r in results},
        'processing_times': {r['model']: r['processing_time'] for r in results},
        'results': results
    }
    
    if len(results) >= 2:
        # Compare symbols between models
        comparison['symbol_differences'] = {}
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                model1 = result1['model']
                model2 = result2['model']
                symbols1 = result1['symbols']
                symbols2 = result2['symbols']
                
                comparison['symbol_differences'][f"{model1}_vs_{model2}"] = {
                    'only_in_first': [s for s in symbols1 if s not in symbols2],
                    'only_in_second': [s for s in symbols2 if s not in symbols1],
                    'common': [s for s in symbols1 if s in symbols2]
                }
    
    return jsonify(comparison)

def create_templates():
    """Create the HTML templates for the web interface"""
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMR Visualizer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        .result-container {
            margin-top: 20px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .symbol-list {
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
        }
        .symbol-item {
            display: inline-block;
            margin: 5px;
            padding: 5px 10px;
            background-color: #e9ecef;
            border-radius: 3px;
        }
        .comparison-table {
            margin-top: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .loading img {
            width: 50px;
            height: 50px;
        }
        .tab-content {
            padding: 20px;
            border: 1px solid #dee2e6;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">Optical Music Recognition (OMR) Visualizer</h1>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="single-tab" data-bs-toggle="tab" data-bs-target="#single" type="button" role="tab" aria-controls="single" aria-selected="true">Single Model</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="compare-tab" data-bs-toggle="tab" data-bs-target="#compare" type="button" role="tab" aria-controls="compare" aria-selected="false">Compare Models</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Single Model Tab -->
            <div class="tab-pane fade show active" id="single" role="tabpanel" aria-labelledby="single-tab">
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="image" class="form-label">Upload Sheet Music Image</label>
                        <input type="file" class="form-control" id="image" name="image" accept="image/*" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="model" class="form-label">Select Model</label>
                        <select class="form-select" id="model" name="model">
                            {% for model in models %}
                            <option value="{{ model }}">{{ model }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Process Image</button>
                </form>
                
                <div class="loading" id="loading-single">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Processing image...</p>
                </div>
                
                <div class="result-container" id="result-container" style="display: none;">
                    <h3>Results</h3>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Original Image</h4>
                            <img id="original-image" class="img-fluid" src="" alt="Original Image">
                        </div>
                        <div class="col-md-6">
                            <h4>Visualization</h4>
                            <img id="visualization" class="img-fluid" src="" alt="Visualization">
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <h4>Recognized Symbols</h4>
                        <p>Model: <span id="result-model"></span></p>
                        <p>Processing Time: <span id="processing-time"></span> seconds</p>
                        <p>Symbol Count: <span id="symbol-count"></span></p>
                        
                        <div class="symbol-list" id="symbol-list"></div>
                    </div>
                </div>
            </div>
            
            <!-- Compare Models Tab -->
            <div class="tab-pane fade" id="compare" role="tabpanel" aria-labelledby="compare-tab">
                <form id="compare-form" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="compare-image" class="form-label">Upload Sheet Music Image</label>
                        <input type="file" class="form-control" id="compare-image" name="image" accept="image/*" required>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Compare All Models</button>
                </form>
                
                <div class="loading" id="loading-compare">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Comparing models...</p>
                </div>
                
                <div class="result-container" id="compare-container" style="display: none;">
                    <h3>Comparison Results</h3>
                    
                    <div class="comparison-table">
                        <h4>Model Performance</h4>
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Symbol Count</th>
                                    <th>Processing Time (s)</th>
                                </tr>
                            </thead>
                            <tbody id="comparison-tbody"></tbody>
                        </table>
                    </div>
                    
                    <div id="model-results"></div>
                    
                    <div id="symbol-differences" style="display: none;">
                        <h4>Symbol Differences</h4>
                        <div id="differences-content"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Single model form submission
            document.getElementById('upload-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const loadingElement = document.getElementById('loading-single');
                const resultContainer = document.getElementById('result-container');
                
                // Show loading indicator
                loadingElement.style.display = 'block';
                resultContainer.style.display = 'none';
                
                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingElement.style.display = 'none';
                    
                    // Display results
                    document.getElementById('original-image').src = 'data:image/png;base64,' + data.original_image;
                    document.getElementById('visualization').src = 'data:image/png;base64,' + data.visualization;
                    document.getElementById('result-model').textContent = data.model;
                    document.getElementById('processing-time').textContent = data.processing_time.toFixed(2);
                    document.getElementById('symbol-count').textContent = data.symbols.length;
                    
                    // Display symbols
                    const symbolList = document.getElementById('symbol-list');
                    symbolList.innerHTML = '';
                    data.symbols.forEach(symbol => {
                        const symbolItem = document.createElement('span');
                        symbolItem.className = 'symbol-item';
                        symbolItem.textContent = symbol;
                        symbolList.appendChild(symbolItem);
                    });
                    
                    resultContainer.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingElement.style.display = 'none';
                    alert('An error occurred while processing the image.');
                });
            });
            
            // Compare models form submission
            document.getElementById('compare-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const loadingElement = document.getElementById('loading-compare');
                const compareContainer = document.getElementById('compare-container');
                
                // Show loading indicator
                loadingElement.style.display = 'block';
                compareContainer.style.display = 'none';
                
                fetch('/compare', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingElement.style.display = 'none';
                    
                    // Display comparison table
                    const comparisonTbody = document.getElementById('comparison-tbody');
                    comparisonTbody.innerHTML = '';
                    
                    data.models.forEach(model => {
                        const row = document.createElement('tr');
                        
                        const modelCell = document.createElement('td');
                        modelCell.textContent = model;
                        row.appendChild(modelCell);
                        
                        const countCell = document.createElement('td');
                        countCell.textContent = data.symbol_counts[model];
                        row.appendChild(countCell);
                        
                        const timeCell = document.createElement('td');
                        timeCell.textContent = data.processing_times[model].toFixed(2);
                        row.appendChild(timeCell);
                        
                        comparisonTbody.appendChild(row);
                    });
                    
                    // Display model results
                    const modelResults = document.getElementById('model-results');
                    modelResults.innerHTML = '';
                    
                    data.results.forEach(result => {
                        const resultDiv = document.createElement('div');
                        resultDiv.className = 'mt-4 p-3 border rounded';
                        
                        const resultTitle = document.createElement('h4');
                        resultTitle.textContent = `Model: ${result.model}`;
                        resultDiv.appendChild(resultTitle);
                        
                        const resultRow = document.createElement('div');
                        resultRow.className = 'row';
                        
                        // Original image column
                        const originalCol = document.createElement('div');
                        originalCol.className = 'col-md-6';
                        
                        const originalTitle = document.createElement('h5');
                        originalTitle.textContent = 'Original Image';
                        originalCol.appendChild(originalTitle);
                        
                        const originalImg = document.createElement('img');
                        originalImg.className = 'img-fluid';
                        originalImg.src = 'data:image/png;base64,' + result.original_image;
                        originalImg.alt = 'Original Image';
                        originalCol.appendChild(originalImg);
                        
                        resultRow.appendChild(originalCol);
                        
                        // Visualization column
                        const visCol = document.createElement('div');
                        visCol.className = 'col-md-6';
                        
                        const visTitle = document.createElement('h5');
                        visTitle.textContent = 'Visualization';
                        visCol.appendChild(visTitle);
                        
                        const visImg = document.createElement('img');
                        visImg.className = 'img-fluid';
                        visImg.src = 'data:image/png;base64,' + result.visualization;
                        visImg.alt = 'Visualization';
                        visCol.appendChild(visImg);
                        
                        resultRow.appendChild(visCol);
                        resultDiv.appendChild(resultRow);
                        
                        // Symbol information
                        const infoDiv = document.createElement('div');
                        infoDiv.className = 'mt-3';
                        
                        const processingTime = document.createElement('p');
                        processingTime.textContent = `Processing Time: ${result.processing_time.toFixed(2)} seconds`;
                        infoDiv.appendChild(processingTime);
                        
                        const symbolCount = document.createElement('p');
                        symbolCount.textContent = `Symbol Count: ${result.symbols.length}`;
                        infoDiv.appendChild(symbolCount);
                        
                        const symbolListTitle = document.createElement('h5');
                        symbolListTitle.textContent = 'Recognized Symbols';
                        infoDiv.appendChild(symbolListTitle);
                        
                        const symbolList = document.createElement('div');
                        symbolList.className = 'symbol-list';
                        
                        result.symbols.forEach(symbol => {
                            const symbolItem = document.createElement('span');
                            symbolItem.className = 'symbol-item';
                            symbolItem.textContent = symbol;
                            symbolList.appendChild(symbolItem);
                        });
                        
                        infoDiv.appendChild(symbolList);
                        resultDiv.appendChild(infoDiv);
                        
                        modelResults.appendChild(resultDiv);
                    });
                    
                    // Display symbol differences if available
                    const symbolDifferences = document.getElementById('symbol-differences');
                    const differencesContent = document.getElementById('differences-content');
                    
                    if (data.symbol_differences) {
                        differencesContent.innerHTML = '';
                        
                        Object.entries(data.symbol_differences).forEach(([comparison, diff]) => {
                            const diffDiv = document.createElement('div');
                            diffDiv.className = 'mt-3 p-3 border rounded';
                            
                            const diffTitle = document.createElement('h5');
                            diffTitle.textContent = comparison;
                            diffDiv.appendChild(diffTitle);
                            
                            // Common symbols
                            const commonTitle = document.createElement('p');
                            commonTitle.innerHTML = `<strong>Common Symbols (${diff.common.length}):</strong>`;
                            diffDiv.appendChild(commonTitle);
                            
                            const commonList = document.createElement('div');
                            commonList.className = 'symbol-list';
                            
                            diff.common.forEach(symbol => {
                                const symbolItem = document.createElement('span');
                                symbolItem.className = 'symbol-item';
                                symbolItem.textContent = symbol;
                                commonList.appendChild(symbolItem);
                            });
                            
                            diffDiv.appendChild(commonList);
                            
                            // Only in first model
                            const firstTitle = document.createElement('p');
                            firstTitle.innerHTML = `<strong>Only in First Model (${diff.only_in_first.length}):</strong>`;
                            diffDiv.appendChild(firstTitle);
                            
                            const firstList = document.createElement('div');
                            firstList.className = 'symbol-list';
                            
                            diff.only_in_first.forEach(symbol => {
                                const symbolItem = document.createElement('span');
                                symbolItem.className = 'symbol-item';
                                symbolItem.style.backgroundColor = '#ffcccc';  // Light red
                                symbolItem.textContent = symbol;
                                firstList.appendChild(symbolItem);
                            });
                            
                            diffDiv.appendChild(firstList);
                            
                            // Only in second model
                            const secondTitle = document.createElement('p');
                            secondTitle.innerHTML = `<strong>Only in Second Model (${diff.only_in_second.length}):</strong>`;
                            diffDiv.appendChild(secondTitle);
                            
                            const secondList = document.createElement('div');
                            secondList.className = 'symbol-list';
                            
                            diff.only_in_second.forEach(symbol => {
                                const symbolItem = document.createElement('span');
                                symbolItem.className = 'symbol-item';
                                symbolItem.style.backgroundColor = '#ccffcc';  // Light green
                                symbolItem.textContent = symbol;
                                secondList.appendChild(symbolItem);
                            });
                            
                            diffDiv.appendChild(secondList);
                            
                            differencesContent.appendChild(diffDiv);
                        });
                        
                        symbolDifferences.style.display = 'block';
                    } else {
                        symbolDifferences.style.display = 'none';
                    }
                    
                    compareContainer.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingElement.style.display = 'none';
                    alert('An error occurred while comparing models.');
                });
            });
        });
    </script>
</body>
</html>
        ''')
    
    # Create directories for static files
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)

def open_browser(port):
    """Open the browser after a short delay"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{port}')

def main():
    parser = argparse.ArgumentParser(
        description='Web-based visualizer for OMR results.')
    
    # Model arguments
    parser.add_argument(
        '-models',
        dest='models',
        type=str,
        nargs='+',
        help='Paths to the trained model meta files. Can specify multiple models.')
    parser.add_argument(
        '-vocabularies',
        dest='vocabularies',
        type=str,
        nargs='+',
        help='Paths to the vocabulary files. Must match the order of models.')
    
    # Server arguments
    parser.add_argument(
        '-port',
        dest='port',
        type=int,
        default=5000,
        help='Port to run the server on.')
    parser.add_argument(
        '-debug',
        dest='debug',
        action='store_true',
        help='Run the server in debug mode.')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.models is None or len(args.models) == 0:
        # Default to using both models if none specified
        args.models = [
            'ml/models/resources/tf-deep-omr/Data/Models/Agnostic Model/agnostic_model.meta',
            'ml/models/resources/tf-deep-omr/Data/Models/Semantic-Model/semantic_model.meta'
        ]
        args.vocabularies = [
            'ml/models/resources/tf-deep-omr/Data/vocabulary_agnostic.txt',
            'ml/models/resources/tf-deep-omr/Data/vocabulary_semantic.txt'
        ]
    elif len(args.models) != len(args.vocabularies):
        parser.error("Number of models must match number of vocabularies")
    
    # Create templates
    create_templates()
    
    # Initialize model testers
    global MODELS
    for model_path, vocab_path in zip(args.models, args.vocabularies):
        model_name = os.path.basename(os.path.dirname(model_path))
        print(f"Loading model: {model_path}")
        MODELS[model_name] = OMRModelTester(model_path, vocab_path)
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()
    
    # Run the server
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
