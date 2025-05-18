import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np
import os
import sys
from datetime import datetime

parser = argparse.ArgumentParser(
    description='Decode a music score image with a trained model (CTC).')
parser.add_argument(
    '-image',
    dest='image',
    type=str,
    required=True,
    help='Path to the input image.')
parser.add_argument(
    '-model',
    dest='model',
    type=str,
    required=True,
    help='Path to the trained model.')
parser.add_argument(
    '-vocabulary',
    dest='voc_file',
    type=str,
    required=True,
    help='Path to the vocabulary file.')
args = parser.parse_args()

# Enable TF1.x compatibility mode
tf.compat.v1.disable_eager_execution()
# Create a new graph and session
graph = tf.compat.v1.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session(graph=graph)

# Read the dictionary
dict_file = open(args.voc_file, 'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

with graph.as_default():
    # Restore weights
    saver = tf.compat.v1.train.import_meta_graph(args.model)
    saver.restore(sess, args.model[:-5])
    
    # Get tensors from the graph
    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.compat.v1.get_collection("logits")[0]

with graph.as_default():
    # Constants that are saved inside the model itself
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
    
    # Setup decoder
    decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)
    
    print(f"Processing image: {args.image}")

    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        sys.exit(1)

    # Read the image
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image '{args.image}'. The file may be corrupted or in an unsupported format.")
        sys.exit(1)

    print(f"Original image shape: {image.shape}")

    # Resize and normalize the image
    try:
        image = ctc_utils.resize(image, HEIGHT)
        image = ctc_utils.normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        print(f"Processed image shape: {image.shape}")
        
        seq_lengths = [image.shape[2] // WIDTH_REDUCTION]  # Integer division
        print(f"Sequence length: {seq_lengths[0]}")
    except Exception as e:
        print(f"Error during image processing: {e}")
        sys.exit(1)
        
    # Run prediction
    print(f"Running prediction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    try:
        prediction = sess.run(decoded,
                            feed_dict={
                                input: image,
                                seq_len: seq_lengths,
                                rnn_keep_prob: 1.0,
                            })
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
        
    # Convert predictions to strings
    try:
        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
        
        # Print results
        print("\nRecognized symbols:")
        if len(str_predictions) > 0 and len(str_predictions[0]) > 0:
            for w in str_predictions[0]:
                if w in int2word:
                    print(int2word[w], end='\t')
                else:
                    print(f"<unknown-{w}>", end='\t')
            print("\n")
            print(f"Total symbols recognized: {len(str_predictions[0])}")
        else:
            print("No symbols were recognized. The model may need more training.")
    except Exception as e:
        print(f"Error processing prediction results: {e}")
        sys.exit(1)
