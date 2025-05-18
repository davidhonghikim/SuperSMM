#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced processor for 'Somewhere Over the Rainbow' sheet music through the OMR pipeline
with improved image preprocessing
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime

# Enable TF1.x compatibility mode
tf.compat.v1.disable_eager_execution()

class OMRProcessor:
    def __init__(self, model_path, vocabulary_path, model_name):
        """Initialize the OMR processor with a model and vocabulary"""
        self.model_path = model_path
        self.vocabulary_path = vocabulary_path
        self.model_name = model_name
        
        # Read the dictionary
        self.int2word = self._load_vocabulary()
        
        # Initialize the model
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session(graph=self.graph)
            
            # Restore weights
            saver = tf.compat.v1.train.import_meta_graph(model_path)
            saver.restore(self.sess, model_path[:-5])
            
            # Get tensors from the graph
            self.input = self.graph.get_tensor_by_name("model_input:0")
            self.seq_len = self.graph.get_tensor_by_name("seq_lengths:0")
            self.rnn_keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
            self.height_tensor = self.graph.get_tensor_by_name("input_height:0")
            self.width_reduction_tensor = self.graph.get_tensor_by_name("width_reduction:0")
            self.logits = tf.compat.v1.get_collection("logits")[0]
            
            # Constants that are saved inside the model itself
            self.WIDTH_REDUCTION, self.HEIGHT = self.sess.run([self.width_reduction_tensor, self.height_tensor])
            
            # Setup decoder
            self.decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(self.logits, self.seq_len)
    
    def _load_vocabulary(self):
        """Load the vocabulary from the vocabulary file"""
        dict_file = open(self.vocabulary_path, 'r')
        dict_list = dict_file.read().splitlines()
        int2word = dict()
        for word in dict_list:
            word_idx = len(int2word)
            int2word[word_idx] = word
        dict_file.close()
        return int2word
    
    @staticmethod
    def normalize(image):
        """Normalize image to range [0, 1]"""
        return (255. - image) / 255.
    
    @staticmethod
    def resize(image, height):
        """Resize image to target height while maintaining aspect ratio"""
        width = int(float(height * image.shape[1]) / image.shape[0])
        sample_img = cv2.resize(image, (width, height))
        return sample_img
    
    @staticmethod
    def preprocess_image(image_path, output_dir=None):
        """Apply advanced preprocessing to improve symbol recognition"""
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Save original for reference
        if output_dir:
            original_path = os.path.join(output_dir, f"original_{os.path.basename(image_path)}")
            cv2.imwrite(original_path, image)
        
        # Apply adaptive thresholding to handle uneven lighting
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Save binary for reference
        if output_dir:
            binary_path = os.path.join(output_dir, f"binary_{os.path.basename(image_path)}")
            cv2.imwrite(binary_path, binary)
        
        # Apply morphological operations to clean up noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Save cleaned for reference
        if output_dir:
            cleaned_path = os.path.join(output_dir, f"cleaned_{os.path.basename(image_path)}")
            cv2.imwrite(cleaned_path, cleaned)
        
        # Extract staff lines
        horizontal_kernel = np.ones((1, 50), np.uint8)
        staff_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Save staff lines for reference
        if output_dir:
            staff_path = os.path.join(output_dir, f"staff_{os.path.basename(image_path)}")
            cv2.imwrite(staff_path, staff_lines)
        
        # Extract music symbols (remove staff lines)
        symbols = cv2.subtract(cleaned, staff_lines)
        
        # Save symbols for reference
        if output_dir:
            symbols_path = os.path.join(output_dir, f"symbols_{os.path.basename(image_path)}")
            cv2.imwrite(symbols_path, symbols)
        
        # Create a composite image with staff lines in gray and symbols in black
        composite = cv2.addWeighted(staff_lines, 0.3, symbols, 1.0, 0)
        
        # Save composite for reference
        if output_dir:
            composite_path = os.path.join(output_dir, f"composite_{os.path.basename(image_path)}")
            cv2.imwrite(composite_path, composite)
        
        # Split the image into staff segments
        staff_height = 150  # Approximate height of a staff in pixels
        staff_segments = []
        
        # Find horizontal projections to locate staff systems
        h_proj = np.sum(staff_lines, axis=1)
        threshold = np.max(h_proj) * 0.5
        
        # Find staff positions
        staff_positions = []
        in_staff = False
        start_pos = 0
        
        for i, proj in enumerate(h_proj):
            if not in_staff and proj > threshold:
                in_staff = True
                start_pos = i
            elif in_staff and proj < threshold:
                in_staff = False
                if i - start_pos > 20:  # Minimum staff height
                    staff_positions.append((start_pos, i))
        
        # Extract each staff with some margin
        margin = 50
        for i, (start, end) in enumerate(staff_positions):
            top = max(0, start - margin)
            bottom = min(image.shape[0], end + margin)
            
            # Extract the staff segment
            segment = composite[top:bottom, :]
            
            # Save staff segment for reference
            if output_dir:
                segment_path = os.path.join(output_dir, f"segment_{i+1}_{os.path.basename(image_path)}")
                cv2.imwrite(segment_path, segment)
            
            staff_segments.append(segment)
        
        return {
            'original': image,
            'binary': binary,
            'cleaned': cleaned,
            'staff_lines': staff_lines,
            'symbols': symbols,
            'composite': composite,
            'segments': staff_segments
        }
    
    @staticmethod
    def sparse_tensor_to_strs(sparse_tensor):
        """Convert sparse tensor to strings"""
        # Handle different formats of sparse_tensor in TF 2.x
        if isinstance(sparse_tensor, tuple) and len(sparse_tensor) == 2:
            # This is the format from tf.nn.ctc_greedy_decoder in TF 2.x
            indices = sparse_tensor[0].indices
            values = sparse_tensor[0].values
            dense_shape = sparse_tensor[0].dense_shape
        else:
            # Original format for backward compatibility
            indices = sparse_tensor[0][0]
            values = sparse_tensor[0][1]
            dense_shape = sparse_tensor[0][2]
    
        strs = [[] for i in range(dense_shape[0])]
    
        string = []
        ptr = 0
        b = 0
    
        for idx in range(len(indices)):
            if indices[idx][0] != b:
                strs[b] = string
                string = []
                b = indices[idx][0]
    
            string.append(values[ptr])
            ptr = ptr + 1
    
        strs[b] = string
        return strs
    
    def process_image(self, image_path, output_dir):
        """Process an image and return the recognized symbols"""
        start_time = time.time()
        
        # Apply advanced preprocessing
        processed = self.preprocess_image(image_path, output_dir)
        if processed is None:
            return {
                "error": f"Failed to preprocess image: {image_path}",
                "symbols": [],
                "processing_time": 0
            }
        
        # Process each staff segment
        all_symbols = []
        segment_results = []
        
        for i, segment in enumerate(processed['segments']):
            # Prepare the segment for the model
            img = self.resize(segment, self.HEIGHT)
            img = self.normalize(img)
            img = np.asarray(img).reshape(1, img.shape[0], img.shape[1], 1)
            
            seq_lengths = [img.shape[2] // self.WIDTH_REDUCTION]  # Integer division
            
            with self.graph.as_default():
                # Run prediction
                prediction = self.sess.run(self.decoded,
                                          feed_dict={
                                              self.input: img,
                                              self.seq_len: seq_lengths,
                                              self.rnn_keep_prob: 1.0,
                                          })
                
                # Convert predictions to strings
                str_predictions = self.sparse_tensor_to_strs(prediction)
                
                # Convert to list of symbols
                symbols = [self.int2word[w] for w in str_predictions[0]]
                all_symbols.extend(symbols)
                
                # Save segment result
                segment_results.append({
                    "segment": i + 1,
                    "symbols": symbols
                })
                
                # Create visualization
                vis_img = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                color = (0, 0, 255)  # Red
                thickness = 1
                
                # Add symbols as text overlay
                y_pos = 20
                for j, symbol in enumerate(symbols):
                    # Position text in a grid-like pattern
                    x_pos = 10 + (j % 5) * (segment.shape[1] // 5)
                    if j > 0 and j % 5 == 0:
                        y_pos += 20
                    
                    cv2.putText(vis_img, symbol, (x_pos, y_pos), 
                                font, font_scale, color, thickness)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f"{self.model_name}_segment_{i+1}_{os.path.basename(image_path)}")
                cv2.imwrite(vis_path, vis_img)
        
        elapsed_time = time.time() - start_time
        
        # Create full page visualization
        img = cv2.imread(image_path)
        vis_img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 255)  # Red
        thickness = 1
        
        # Add symbols as text overlay
        y_pos = 30
        for i, symbol in enumerate(all_symbols):
            # Position text in a grid-like pattern
            x_pos = 10 + (i % 8) * (img.shape[1] // 8)
            if i > 0 and i % 8 == 0:
                y_pos += 25
            
            cv2.putText(vis_img, symbol, (x_pos, y_pos), 
                        font, font_scale, color, thickness)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{self.model_name}_{os.path.basename(image_path)}")
        cv2.imwrite(vis_path, vis_img)
        
        # Save symbols to text file
        txt_path = os.path.join(output_dir, f"{self.model_name}_{os.path.splitext(os.path.basename(image_path))[0]}.txt")
        with open(txt_path, 'w') as f:
            f.write('\t'.join(all_symbols))
        
        # Save detailed JSON result
        json_path = os.path.join(output_dir, f"{self.model_name}_{os.path.splitext(os.path.basename(image_path))[0]}.json")
        result = {
            "image_path": image_path,
            "model": self.model_name,
            "symbols": all_symbols,
            "segments": segment_results,
            "processing_time": elapsed_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced processor for Somewhere Over the Rainbow through the OMR pipeline')
    
    parser.add_argument(
        '-output',
        dest='output_dir',
        type=str,
        default='output/rainbow_enhanced',
        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define models
    models = [
        {
            'name': 'Agnostic',
            'meta': '/ml/models/resources/tf-deep-omr/Data/Models/Agnostic-Model/agnostic_model.meta',
            'vocabulary': '/ml/models/resources/tf-deep-omr/Data/vocabulary_agnostic.txt'
        },
        {
            'name': 'Semantic',
            'meta': '/ml/models/resources/tf-deep-omr/Data/Models/Semantic-Model/semantic_model.meta',
            'vocabulary': '/ml/models/resources/tf-deep-omr/Data/vocabulary_semantic.txt'
        }
    ]
    
    # Define images
    images = [
        '/ml/models/resources/tf-deep-omr/Data/Somewhere_Over_the_Rainbow_page_1.png',
        '/ml/models/resources/tf-deep-omr/Data/Somewhere_Over_the_Rainbow_page_2.png'
    ]
    
    # Process each image with each model
    all_results = []
    
    for model in models:
        print(f"Processing with {model['name']} model...")
        processor = OMRProcessor(model['meta'], model['vocabulary'], model['name'])
        
        for image_path in images:
            print(f"  Processing {os.path.basename(image_path)}...")
            result = processor.process_image(image_path, args.output_dir)
            all_results.append(result)
            print(f"  Done. Found {len(result['symbols'])} symbols in {result['processing_time']:.2f} seconds.")
    
    # Generate summary report
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate HTML report
    html_path = os.path.join(args.output_dir, "report.html")
    with open(html_path, 'w') as f:
        f.write(f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Somewhere Over the Rainbow - Enhanced OMR Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-container {{
            margin-top: 20px;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .model-results {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        .model-card {{
            flex: 1;
            min-width: 300px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
        }}
        .model-header {{
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }}
        .symbol-list {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .symbol {{
            display: inline-block;
            margin: 3px;
            padding: 2px 8px;
            background-color: #e9ecef;
            border-radius: 3px;
        }}
        .preprocessing {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }}
        .preprocessing-step {{
            flex: 1;
            min-width: 200px;
            text-align: center;
        }}
        .preprocessing-step img {{
            max-width: 100%;
            max-height: 200px;
            border: 1px solid #ddd;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Somewhere Over the Rainbow - Enhanced OMR Results</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="card">
            <h2>Processing Summary</h2>
            <p>This report shows the results of processing "Somewhere Over the Rainbow" sheet music through an enhanced OMR pipeline with advanced image preprocessing.</p>
            
            <h3>Image Preprocessing</h3>
            <p>The following preprocessing steps were applied to improve symbol recognition:</p>
            <ol>
                <li>Adaptive thresholding to handle uneven lighting</li>
                <li>Morphological operations to clean up noise</li>
                <li>Staff line extraction</li>
                <li>Symbol extraction (removing staff lines)</li>
                <li>Staff segmentation for improved recognition</li>
            </ol>
            
            <div class="preprocessing">
                <div class="preprocessing-step">
                    <h4>Original</h4>
                    <img src="original_Somewhere_Over_the_Rainbow_page_1.png" alt="Original">
                </div>
                <div class="preprocessing-step">
                    <h4>Binary</h4>
                    <img src="binary_Somewhere_Over_the_Rainbow_page_1.png" alt="Binary">
                </div>
                <div class="preprocessing-step">
                    <h4>Cleaned</h4>
                    <img src="cleaned_Somewhere_Over_the_Rainbow_page_1.png" alt="Cleaned">
                </div>
                <div class="preprocessing-step">
                    <h4>Staff Lines</h4>
                    <img src="staff_Somewhere_Over_the_Rainbow_page_1.png" alt="Staff Lines">
                </div>
                <div class="preprocessing-step">
                    <h4>Symbols</h4>
                    <img src="symbols_Somewhere_Over_the_Rainbow_page_1.png" alt="Symbols">
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Page 1 Results</h2>
            
            <div class="model-results">
                <div class="model-card">
                    <div class="model-header">Agnostic Model</div>
                    <p>Processing time: {next((r['processing_time'] for r in all_results if r['model'] == 'Agnostic' and 'page_1' in r['image_path']), 0):.2f} seconds</p>
                    <p>Symbols found: {len(next((r['symbols'] for r in all_results if r['model'] == 'Agnostic' and 'page_1' in r['image_path']), []))}</p>
                    <div class="image-container">
                        <img src="Agnostic_Somewhere_Over_the_Rainbow_page_1.png" alt="Agnostic Model - Page 1">
                    </div>
                    <h4>Recognized Symbols:</h4>
                    <div class="symbol-list">
                        {' '.join(f'<span class="symbol">{s}</span>' for s in next((r['symbols'] for r in all_results if r['model'] == 'Agnostic' and 'page_1' in r['image_path']), []))}
                    </div>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <p>Processing time: {next((r['processing_time'] for r in all_results if r['model'] == 'Semantic' and 'page_1' in r['image_path']), 0):.2f} seconds</p>
                    <p>Symbols found: {len(next((r['symbols'] for r in all_results if r['model'] == 'Semantic' and 'page_1' in r['image_path']), []))}</p>
                    <div class="image-container">
                        <img src="Semantic_Somewhere_Over_the_Rainbow_page_1.png" alt="Semantic Model - Page 1">
                    </div>
                    <h4>Recognized Symbols:</h4>
                    <div class="symbol-list">
                        {' '.join(f'<span class="symbol">{s}</span>' for s in next((r['symbols'] for r in all_results if r['model'] == 'Semantic' and 'page_1' in r['image_path']), []))}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Page 2 Results</h2>
            
            <div class="model-results">
                <div class="model-card">
                    <div class="model-header">Agnostic Model</div>
                    <p>Processing time: {next((r['processing_time'] for r in all_results if r['model'] == 'Agnostic' and 'page_2' in r['image_path']), 0):.2f} seconds</p>
                    <p>Symbols found: {len(next((r['symbols'] for r in all_results if r['model'] == 'Agnostic' and 'page_2' in r['image_path']), []))}</p>
                    <div class="image-container">
                        <img src="Agnostic_Somewhere_Over_the_Rainbow_page_2.png" alt="Agnostic Model - Page 2">
                    </div>
                    <h4>Recognized Symbols:</h4>
                    <div class="symbol-list">
                        {' '.join(f'<span class="symbol">{s}</span>' for s in next((r['symbols'] for r in all_results if r['model'] == 'Agnostic' and 'page_2' in r['image_path']), []))}
                    </div>
                </div>
                
                <div class="model-card">
                    <div class="model-header">Semantic Model</div>
                    <p>Processing time: {next((r['processing_time'] for r in all_results if r['model'] == 'Semantic' and 'page_2' in r['image_path']), 0):.2f} seconds</p>
                    <p>Symbols found: {len(next((r['symbols'] for r in all_results if r['model'] == 'Semantic' and 'page_2' in r['image_path']), []))}</p>
                    <div class="image-container">
                        <img src="Semantic_Somewhere_Over_the_Rainbow_page_2.png" alt="Semantic Model - Page 2">
                    </div>
                    <h4>Recognized Symbols:</h4>
                    <div class="symbol-list">
                        {' '.join(f'<span class="symbol">{s}</span>' for s in next((r['symbols'] for r in all_results if r['model'] == 'Semantic' and 'page_2' in r['image_path']), []))}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Staff Segments</h2>
            <p>The sheet music was segmented into individual staff systems for improved recognition:</p>
            
            <h3>Page 1 Segments</h3>
            <div class="preprocessing">
                {' '.join(f'<div class="preprocessing-step"><h4>Segment {i+1}</h4><img src="segment_{i+1}_Somewhere_Over_the_Rainbow_page_1.png" alt="Segment {i+1}"></div>' for i in range(len(next((r['segments'] for r in all_results if r['model'] == 'Agnostic' and 'page_1' in r['image_path']), []))))}
            </div>
            
            <h3>Page 2 Segments</h3>
            <div class="preprocessing">
                {' '.join(f'<div class="preprocessing-step"><h4>Segment {i+1}</h4><img src="segment_{i+1}_Somewhere_Over_the_Rainbow_page_2.png" alt="Segment {i+1}"></div>' for i in range(len(next((r['segments'] for r in all_results if r['model'] == 'Agnostic' and 'page_2' in r['image_path']), []))))}
            </div>
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
        
        <div class="footer">
            <p>Generated by Enhanced OMR Processing System</p>
        </div>
    </div>
</body>
</html>
        ''')
    
    print("\nProcessing completed successfully!")
    print(f"Results saved to {args.output_dir}/")
    print(f"HTML report: {html_path}")
    print(f"Summary JSON: {summary_path}")

if __name__ == '__main__':
    main()
