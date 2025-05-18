#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process 'Somewhere Over the Rainbow' sheet music through the OMR pipeline
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
        
        # Process the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {
                "error": f"Failed to load image: {image_path}",
                "symbols": [],
                "processing_time": 0
            }
            
        image = self.resize(image, self.HEIGHT)
        image = self.normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        
        seq_lengths = [image.shape[2] // self.WIDTH_REDUCTION]  # Integer division
        
        with self.graph.as_default():
            # Run prediction
            prediction = self.sess.run(self.decoded,
                                      feed_dict={
                                          self.input: image,
                                          self.seq_len: seq_lengths,
                                          self.rnn_keep_prob: 1.0,
                                      })
            
            # Convert predictions to strings
            str_predictions = self.sparse_tensor_to_strs(prediction)
            
            # Convert to list of symbols
            symbols = [self.int2word[w] for w in str_predictions[0]]
            
            elapsed_time = time.time() - start_time
            
            # Create visualization
            img = cv2.imread(image_path)
            vis_img = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 255)  # Red
            thickness = 1
            
            # Add symbols as text overlay
            y_pos = 20
            for i, symbol in enumerate(symbols):
                # Position text in a grid-like pattern
                x_pos = 10 + (i % 5) * (img.shape[1] // 5)
                if i > 0 and i % 5 == 0:
                    y_pos += 20
                
                cv2.putText(vis_img, symbol, (x_pos, y_pos), 
                            font, font_scale, color, thickness)
            
            # Save visualization
            image_name = os.path.basename(image_path)
            vis_path = os.path.join(output_dir, f"{self.model_name}_{image_name}")
            cv2.imwrite(vis_path, vis_img)
            
            # Save symbols to text file
            txt_path = os.path.join(output_dir, f"{self.model_name}_{os.path.splitext(image_name)[0]}.txt")
            with open(txt_path, 'w') as f:
                f.write('\t'.join(symbols))
            
            # Save detailed JSON result
            json_path = os.path.join(output_dir, f"{self.model_name}_{os.path.splitext(image_name)[0]}.json")
            result = {
                "image_path": image_path,
                "model": self.model_name,
                "symbols": symbols,
                "processing_time": elapsed_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result

def main():
    parser = argparse.ArgumentParser(
        description='Process Somewhere Over the Rainbow through the OMR pipeline')
    
    parser.add_argument(
        '-output',
        dest='output_dir',
        type=str,
        default='output/rainbow',
        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define models
    models = [
        {
            'name': 'Agnostic',
            'meta': 'Data/Models/Agnostic Model/agnostic_model.meta',
            'vocabulary': 'Data/vocabulary_agnostic.txt'
        },
        {
            'name': 'Semantic',
            'meta': 'Data/Models/Semantic-Model/semantic_model.meta',
            'vocabulary': 'Data/vocabulary_semantic.txt'
        }
    ]
    
    # Define images
    images = [
        'Data/Somewhere_Over_the_Rainbow_page_1.png',
        'Data/Somewhere_Over_the_Rainbow_page_2.png'
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
    <title>Somewhere Over the Rainbow - OMR Results</title>
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
        <h1>Somewhere Over the Rainbow - OMR Results</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="card">
            <h2>Page 1</h2>
            <div class="image-container">
                <h3>Original Image</h3>
                <img src="../Somewhere_Over_the_Rainbow_page_1.png" alt="Page 1">
            </div>
            
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
            <h2>Page 2</h2>
            <div class="image-container">
                <h3>Original Image</h3>
                <img src="../Somewhere_Over_the_Rainbow_page_2.png" alt="Page 2">
            </div>
            
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
            <p>Generated by OMR Processing System</p>
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
