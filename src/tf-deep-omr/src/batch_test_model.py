#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch testing script for the OMR model using TensorFlow 2.x
This script can process multiple images and compare results between different models
"""

import argparse
import tensorflow as tf
import cv2
import numpy as np
import os
import time
import glob
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Enable TF1.x compatibility mode
tf.compat.v1.disable_eager_execution()

class OMRModelTester:
    def __init__(self, model_path, vocabulary_path):
        """Initialize the OMR model tester with a model and vocabulary"""
        self.model_path = model_path
        self.vocabulary_path = vocabulary_path
        self.model_name = os.path.basename(os.path.dirname(model_path))
        
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
    
    def process_image(self, image_path):
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
            
            return {
                "image_path": image_path,
                "model": self.model_name,
                "symbols": symbols,
                "processing_time": elapsed_time
            }

def process_directory(model_testers, directory, output_file=None, max_workers=4):
    """Process all images in a directory with multiple models"""
    # Find all image files in the directory
    image_files = []
    for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']:
        image_files.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'**/*.{ext}'), recursive=True))
    
    image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
    
    if not image_files:
        print(f"No image files found in {directory}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    
    # Process each image with each model
    for model_tester in model_testers:
        print(f"Processing images with model: {model_tester.model_name}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(model_tester.process_image, img_path) for img_path in image_files]
            
            # Use tqdm for progress bar
            model_results = []
            for future in tqdm(futures, total=len(image_files), desc=f"Model {model_tester.model_name}"):
                model_results.append(future.result())
            
            results.extend(model_results)
    
    # Save results to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def compare_models(results, output_file=None):
    """Compare results between different models"""
    # Group results by image
    image_results = {}
    for result in results:
        if 'error' in result:
            continue
        
        image_path = result['image_path']
        if image_path not in image_results:
            image_results[image_path] = {}
        
        image_results[image_path][result['model']] = {
            'symbols': result['symbols'],
            'processing_time': result['processing_time']
        }
    
    # Create comparison report
    comparison = []
    for image_path, models in image_results.items():
        if len(models) < 2:
            continue  # Skip if only one model processed this image
        
        model_names = list(models.keys())
        comparison_item = {
            'image_path': image_path,
            'models': model_names,
            'symbol_counts': {model: len(data['symbols']) for model, data in models.items()},
            'processing_times': {model: data['processing_time'] for model, data in models.items()},
            'symbol_differences': {}
        }
        
        # Compare symbols between models
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                symbols1 = models[model1]['symbols']
                symbols2 = models[model2]['symbols']
                
                # Calculate differences
                comparison_item['symbol_differences'][f"{model1}_vs_{model2}"] = {
                    'only_in_first': [s for s in symbols1 if s not in symbols2],
                    'only_in_second': [s for s in symbols2 if s not in symbols1],
                    'common': [s for s in symbols1 if s in symbols2]
                }
        
        comparison.append(comparison_item)
    
    # Save comparison to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to {output_file}")
    
    return comparison

def process_single_image(model_testers, image_path):
    """Process a single image with multiple models and display results"""
    results = []
    
    for model_tester in model_testers:
        print(f"\nProcessing with model: {model_tester.model_name}")
        result = model_tester.process_image(image_path)
        results.append(result)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("\nRecognized symbols:")
            for symbol in result['symbols']:
                print(symbol, end='\t')
            print(f"\n\nProcessing completed in {result['processing_time']:.2f} seconds")
    
    # If we have multiple models, show a simple comparison
    if len(model_testers) > 1:
        print("\n\nModel Comparison:")
        for result in results:
            if 'error' in result:
                continue
            print(f"{result['model']}: {len(result['symbols'])} symbols in {result['processing_time']:.2f} seconds")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Test OMR models on images.')
    
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
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-image',
        dest='image',
        type=str,
        help='Path to a single input image.')
    group.add_argument(
        '-directory',
        dest='directory',
        type=str,
        help='Path to a directory of images to process.')
    
    # Output arguments
    parser.add_argument(
        '-output',
        dest='output',
        type=str,
        help='Path to save the results JSON file.')
    parser.add_argument(
        '-comparison',
        dest='comparison',
        type=str,
        help='Path to save the model comparison JSON file.')
    
    # Processing arguments
    parser.add_argument(
        '-workers',
        dest='workers',
        type=int,
        default=4,
        help='Number of worker threads for batch processing.')
    
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
    
    # Initialize model testers
    model_testers = []
    for model_path, vocab_path in zip(args.models, args.vocabularies):
        print(f"Loading model: {model_path}")
        model_testers.append(OMRModelTester(model_path, vocab_path))
    
    # Process images
    if args.image:
        # Process a single image
        results = process_single_image(model_testers, args.image)
    else:
        # Process a directory of images
        results = process_directory(model_testers, args.directory, args.output, args.workers)
        
        # Compare models if we have multiple
        if len(model_testers) > 1:
            compare_models(results, args.comparison)

if __name__ == "__main__":
    main()
