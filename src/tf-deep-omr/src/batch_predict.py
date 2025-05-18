#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch prediction script for the OMR model.
This script processes multiple images and evaluates the model's performance.
"""

import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np
import os
import sys
import time
import glob
from datetime import datetime
import json

def load_model(model_path, vocabulary_path):
    """
    Load the trained model and vocabulary.
    
    Args:
        model_path (str): Path to the trained model
        vocabulary_path (str): Path to the vocabulary file
        
    Returns:
        tuple: (session, input_tensor, seq_len_tensor, rnn_keep_prob_tensor, 
                decoded_tensor, height, width_reduction, int2word)
    """
    print(f"Loading model from {model_path}")
    
    # Read the dictionary
    dict_file = open(vocabulary_path, 'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()
    
    print(f"Loaded vocabulary with {len(int2word)} symbols")
    
    # Enable TF1.x compatibility mode
    tf.compat.v1.disable_eager_execution()
    
    # Create a new graph and session
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        
        # Restore weights
        saver = tf.compat.v1.train.import_meta_graph(model_path)
        saver.restore(sess, model_path[:-5])
        
        # Get tensors from the graph
        input_tensor = graph.get_tensor_by_name("model_input:0")
        seq_len_tensor = graph.get_tensor_by_name("seq_lengths:0")
        rnn_keep_prob_tensor = graph.get_tensor_by_name("keep_prob:0")
        height_tensor = graph.get_tensor_by_name("input_height:0")
        width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        logits = tf.compat.v1.get_collection("logits")[0]
        
        # Setup decoder
        decoded_tensor, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len_tensor)
        
        # Constants that are saved inside the model itself
        WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
    
    return (sess, input_tensor, seq_len_tensor, rnn_keep_prob_tensor, 
            decoded_tensor, HEIGHT, WIDTH_REDUCTION, int2word, graph)

def process_image(image_path, height):
    """
    Process an image for prediction.
    
    Args:
        image_path (str): Path to the image file
        height (int): Target height for resizing
        
    Returns:
        tuple: (processed_image, original_image)
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return None, None
    
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image '{image_path}'.")
        return None, None
    
    # Store original image for reference
    original_image = image.copy()
    
    # Resize and normalize the image
    try:
        image = ctc_utils.resize(image, height)
        image = ctc_utils.normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        return image, original_image
    except Exception as e:
        print(f"Error during image processing: {e}")
        return None, None

def predict_single_image(sess, image, input_tensor, seq_len_tensor, 
                        rnn_keep_prob_tensor, decoded_tensor, width_reduction, int2word):
    """
    Run prediction on a single processed image.
    
    Args:
        sess: TensorFlow session
        image: Processed image
        input_tensor: Input tensor
        seq_len_tensor: Sequence length tensor
        rnn_keep_prob_tensor: RNN keep probability tensor
        decoded_tensor: Decoded tensor
        width_reduction: Width reduction factor
        int2word: Dictionary mapping indices to symbols
        
    Returns:
        list: Recognized symbols
    """
    seq_lengths = [image.shape[2] // width_reduction]
    
    try:
        prediction = sess.run(decoded_tensor,
                            feed_dict={
                                input_tensor: image,
                                seq_len_tensor: seq_lengths,
                                rnn_keep_prob_tensor: 1.0,
                            })
        
        # Convert predictions to strings
        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
        
        if len(str_predictions) > 0 and len(str_predictions[0]) > 0:
            symbols = [int2word.get(w, f"<unknown-{w}>") for w in str_predictions[0]]
            return symbols
        else:
            return []
    except Exception as e:
        print(f"Error during prediction: {e}")
        return []

def batch_predict(model_path, vocabulary_path, image_dir, output_file=None):
    """
    Run prediction on multiple images and evaluate performance.
    
    Args:
        model_path (str): Path to the trained model
        vocabulary_path (str): Path to the vocabulary file
        image_dir (str): Directory containing images to process
        output_file (str, optional): Path to save results
    """
    # Load model
    (sess, input_tensor, seq_len_tensor, rnn_keep_prob_tensor, 
     decoded_tensor, height, width_reduction, int2word, graph) = load_model(model_path, vocabulary_path)
    
    # Find all images in the directory
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    start_time = time.time()
    
    for i, image_path in enumerate(image_files[:100]):  # Limit to 100 images for testing
        print(f"Processing image {i+1}/{min(len(image_files), 100)}: {image_path}")
        
        # Process the image
        processed_image, original_image = process_image(image_path, height)
        if processed_image is None:
            continue
        
        # Run prediction
        with graph.as_default():
            symbols = predict_single_image(
                sess, processed_image, input_tensor, seq_len_tensor,
                rnn_keep_prob_tensor, decoded_tensor, width_reduction, int2word
            )
        
        # Store results
        result = {
            'image_path': image_path,
            'symbols': symbols,
            'symbol_count': len(symbols)
        }
        results.append(result)
        
        print(f"  Recognized {len(symbols)} symbols")
        if symbols:
            print(f"  First few symbols: {' '.join(symbols[:10])}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    avg_time = total_time / len(results) if results else 0
    
    summary = {
        'total_images': len(results),
        'total_time': total_time,
        'avg_time_per_image': avg_time,
        'images_with_symbols': sum(1 for r in results if r['symbols']),
        'avg_symbols_per_image': sum(r['symbol_count'] for r in results) / len(results) if results else 0
    }
    
    print("\nBatch Prediction Summary:")
    print(f"Total images processed: {summary['total_images']}")
    print(f"Total processing time: {summary['total_time']:.2f} seconds")
    print(f"Average time per image: {summary['avg_time_per_image']:.2f} seconds")
    print(f"Images with recognized symbols: {summary['images_with_symbols']}")
    print(f"Average symbols per image: {summary['avg_symbols_per_image']:.2f}")
    
    # Save results to file if requested
    if output_file:
        output_data = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch predict music scores with a trained model.')
    parser.add_argument(
        '-model', dest='model', type=str, required=True,
        help='Path to the trained model.')
    parser.add_argument(
        '-vocabulary', dest='voc_file', type=str, required=True,
        help='Path to the vocabulary file.')
    parser.add_argument(
        '-image_dir', dest='image_dir', type=str, required=True,
        help='Directory containing images to process.')
    parser.add_argument(
        '-output', dest='output_file', type=str, default=None,
        help='Path to save results (optional).')
    args = parser.parse_args()
    
    batch_predict(args.model, args.voc_file, args.image_dir, args.output_file)

if __name__ == "__main__":
    main()
