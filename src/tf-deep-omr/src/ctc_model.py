import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow.compat.v1 as tf1
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Bidirectional

def leaky_relu(features, alpha=0.2, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)

#
# params["height"] = height of the input image
# params["width"] = width of the input image

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [[3, 3], [3, 3], [3, 3], [3, 3]]
    params['conv_pooling_size'] = [[2, 2], [2, 2], [2, 2], [2, 2]]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params

def ctc_crnn(params):
    # TODO Assert parameters

    input = tf1.placeholder(shape=(None,
                                  params['img_height'],
                                  params['img_width'],
                                  params['img_channels']),  # [batch, height, width, channels]
                           dtype=tf.float32,
                           name='model_input')

    input_shape = tf.shape(input)

    width_reduction = 1
    height_reduction = 1

    # Convolutional blocks
    x = input
    for i in range(params['conv_blocks']):
        # Use Keras layers instead of tf.layers
        conv = layers.Conv2D(
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)
        x = conv(x)

        batch_norm = layers.BatchNormalization()
        x = batch_norm(x)
        x = leaky_relu(x)

        pooling = layers.MaxPooling2D(
            pool_size=params['conv_pooling_size'][i],
            strides=params['conv_pooling_size'][i])
        x = pooling(x)

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    # Prepare output of conv block for recurrent blocks
    features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] / width_reduction
    features = tf.reshape(features, tf.stack(
        [tf.cast(feature_width, 'int32'), input_shape[0], tf.cast(feature_dim, 'int32')]))  # -> [width, batch, features]

    tf1.constant(params['img_height'], name='input_height')
    tf1.constant(width_reduction, name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf1.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    # Use TF 2.x compatible RNN with proper time_major handling
    # Convert from time_major to batch_major for Keras
    x = tf.transpose(features, [1, 0, 2])
    
    for i in range(rnn_hidden_layers):
        # Apply dropout before the LSTM layer
        x = tf.nn.dropout(x, rate=1-rnn_keep_prob)
        # Use Bidirectional LSTM from Keras
        lstm_layer = Bidirectional(
            LSTM(rnn_hidden_units, return_sequences=True),
            merge_mode='concat'
        )
        x = lstm_layer(x)
    
    # Convert back to time_major for the rest of the network
    x = tf.transpose(x, [1, 0, 2])
    
    rnn_outputs = x
    
    # Use Keras Dense layer
    dense = layers.Dense(params['vocabulary_size'] + 1, activation=None)
    logits = dense(rnn_outputs)

    tf.compat.v1.add_to_collection("logits", logits)  # for restoring purposes

    # CTC Loss computation
    seq_len = tf1.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf1.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    loss = tf.reduce_mean(ctc_loss)

    # CTC decoding with safety checks
    # Ensure sequence length is at least 1 to prevent empty inputs error
    safe_seq_len = tf.maximum(seq_len, tf.ones_like(seq_len))
    decoded, log_prob = tf.compat.v1.nn.ctc_greedy_decoder(logits, safe_seq_len)
    # Alternative: decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, safe_seq_len, beam_width=50, top_paths=1, merge_repeated=True)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob
