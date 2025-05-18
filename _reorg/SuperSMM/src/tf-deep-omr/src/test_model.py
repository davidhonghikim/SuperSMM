#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the OMR model using TensorFlow 2.x
"""

import argparse
import tensorflow as tf
import cv2
import numpy as np
import os
import time
from datetime import datetime

# Enable TF1.x compatibility mode
tf.compat.v1.disable_eager_execution()

def normalize(image):
    """Normalize image to range [0, 1]"""
    return (255. - image) / 255.

def resize(image, height):
    """Resize image to target height while maintaining aspect ratio"""
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img

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

def test_model(model_path, vocabulary_path, image_path):
    """Test the OMR model on a given image"""
    # Read the dictionary
    dict_file = open(vocabulary_path, 'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    # Create a new graph and session
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph)
        
        # Restore weights
        saver = tf.compat.v1.train.import_meta_graph(model_path)
        saver.restore(sess, model_path[:-5])
        
        # Get tensors from the graph
        input = graph.get_tensor_by_name("model_input:0")
        seq_len = graph.get_tensor_by_name("seq_lengths:0")
        rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        height_tensor = graph.get_tensor_by_name("input_height:0")
        width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        logits = tf.compat.v1.get_collection("logits")[0]
        
        # Constants that are saved inside the model itself
        WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
        
        # Setup decoder
        decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)
        
        # Process the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        image = resize(image, HEIGHT)
        image = normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        
        seq_lengths = [image.shape[2] // WIDTH_REDUCTION]  # Integer division
        
        # Run prediction
        prediction = sess.run(decoded,
                              feed_dict={
                                  input: image,
                                  seq_len: seq_lengths,
                                  rnn_keep_prob: 1.0,
                              })
        
        # Convert predictions to strings
        str_predictions = sparse_tensor_to_strs(prediction)
        
        # Print results
        print("\nRecognized symbols:")
        for w in str_predictions[0]:
            print(int2word[w], end='\t')
        print("\n")

def main():
    parser = argparse.ArgumentParser(
        description='Test a trained OMR model on an image.')
    parser.add_argument(
        '-model',
        dest='model',
        type=str,
        required=True,
        help='Path to the trained model meta file.')
    parser.add_argument(
        '-vocabulary',
        dest='voc_file',
        type=str,
        required=True,
        help='Path to the vocabulary file.')
    parser.add_argument(
        '-image',
        dest='image',
        type=str,
        required=True,
        help='Path to the input image.')
    
    args = parser.parse_args()
    
    start_time = time.time()
    test_model(args.model, args.voc_file, args.image)
    elapsed_time = time.time() - start_time
    
    print(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
