import os
import sys
import csv
from pathlib import Path
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras.preprocessing import sequence
import numpy as np
import msgpack
from io import BytesIO
import logging
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
# TODO Use this or get rid of it.
from imblearn.over_sampling import (ADASYN, RandomOverSampler, SMOTE)
import requests
import json


print('TensorFlow version:', tf.__version__)

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", None, "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If "
    "not specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project."
    " If not specified, we will attempt to automatically detect the GCE project from"
    " metadata.")

# Store CLI args as flags in tf.
tf.flags.DEFINE_string("exp_name", "Default-Experiment",
                       "Name of the experiment to which the run belongs.")
tf.flags.DEFINE_integer("exp_num", None,
                        "Number of the experiment to which the run belongs.")
tf.flags.DEFINE_string("data_dir", "",
                       "Path to the data in messagepack bin format.")
tf.flags.DEFINE_string("embeddings_dir", "",
                       "Path to the word embeddings in numpy format.")
tf.flags.DEFINE_string("lexicons_path", "lexicons/lex_emb.bin",
                       "Path to lexicons config file.")
tf.flags.DEFINE_string("decoding_dic_path", "",
                       "Path to id2word dictionary in messagepack bin format.")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_bool("save_model", False, "Export trained estimator.")
tf.flags.DEFINE_string("export_dir_base", None,
                       "Base path to which exported model will be saved.")
tf.flags.DEFINE_string("csv_path", "", "Path to which to write CSV of results.")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", False, "Do some predictions at the end.")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training epochs.")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout rate.")
tf.flags.DEFINE_integer("max_doc_length", 500, "Maximum num of words per document.")
tf.flags.DEFINE_bool("random_init_embeddings", False,
                     "Use random uniform data to initialize embeddings.")
tf.flags.DEFINE_float("dev_set_size", 0.10,
                      "Size of the development set as fraction of total.")

# Notifications.
tf.flags.DEFINE_bool("use_slack", False,
                     "Use slack to receive notifications when run is done.")
tf.flags.DEFINE_string("slack_webhook_url", "", "Slack app webhook.")

FLAGS = tf.flags.FLAGS

# Read in embeddings.
logging.info('Loading data.')
f = BytesIO(file_io.read_file_to_string(FLAGS.embeddings_dir, binary_mode=True))
vocab = np.load(f)

# Add one additional row to embeddings for unknown words.
# Add additional word embedding for unknown words.
vocab = np.concatenate((vocab, np.random.rand(1, vocab.shape[1])))

# Print size of vocab to confirm size.
print('len(vocab):', len(vocab))

# Read in lexicon embeddings.
f = BytesIO(file_io.read_file_to_string(FLAGS.lexicons_path, binary_mode=True))
lex_emb = np.load(f)

# Print size of lexicons to confirm size.
print('len(lex_emb):', len(lex_emb))

# Read in id2word for decoding the encoded examples.
f = BytesIO(file_io.read_file_to_string(FLAGS.decoding_dic_path, binary_mode=True))
id2word = msgpack.unpack(f, raw=False)

# Visual check to make sure id2word dict is read in properly.
for i in range(0, 10):
    print(i, id2word.get(i))

# Read in features and labels.
f = BytesIO(file_io.read_file_to_string(FLAGS.data_dir, binary_mode=True))
data = msgpack.unpack(f, raw=False)


# Decodes encoded text using the id2word dict.
def indexes_to_text(indexes):
    found_indexes_list = []
    not_found_indexes_list = []

    for index in indexes:
        if id2word.get(index) is not None:
            found_indexes_list.append(id2word.get(index))
        else:
            not_found_indexes_list.append(index)

    print('Indexes not found:', not_found_indexes_list)

    return ' '.join(found_indexes_list)


def prepare_data(raw_data):
    # Convert data to numpy arrays.
    logging.info('Converting data to arrays.')

    # For keeping number of words in longest document.
    max_words = 0

    # Create lists to store docs and labels.
    docs = []
    labels = []

    # Iterate over data to build lists of docs and labels.
    num_docs = len(raw_data)
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i + 1, num_docs))
        sys.stdout.flush()

        # Get index of document.
        doc = raw_data[i]['idx']

        # Retrieve document from saved data and cast to list.
        doc = [item for sublist in doc for item in sublist]

        # Add document to docs array.
        docs.append(doc)

        # Add label to label array at same index.
        labels.append(raw_data[i]['label'])

        # Track maximum number of words in document.
        if len(doc) > max_words:
            max_words = len(doc)

    del raw_data
    print()

    # TODO Still needed?
    # Label encoder.
    #   Encode labels with value between 0 and n_classes-1,
    #   so for example 1 to 5 star ratings become 0 to 4.
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # TODO Still needed?
    # Binarize labels in one-vs-all fashion if three or more classes.   
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(docs, y_bin,
                                                        test_size=FLAGS.dev_set_size)

    # Decode and print first train example.
    print('Decoded first training doc:', indexes_to_text(x_train[0]))

    # Char used as padding elements.
    pad_id = 0

    # Pads all docs to the length of the longest doc using the pad_id char.
    x_train_padded = sequence.pad_sequences(x_train,
                                            maxlen=FLAGS.max_doc_length,
                                            truncating='post',
                                            padding='post',
                                            value=pad_id)

    x_test_padded = sequence.pad_sequences(x_test,
                                           maxlen=FLAGS.max_doc_length,
                                           truncating='post',
                                           padding='post',
                                           value=pad_id)

    print('Unpadded length of first training doc:\t', len(x_train[0]))
    print('Unpadded length of second training doc:\t', len(x_train[1]))
    print('Padded len of first doc:\t', len(x_train_padded[0]))
    print('Padded len of second doc:\t', len(x_train_padded[1]))
    print('x_train shape:\t\t\t', x_train_padded.shape)
    print('x_test shape:\t\t\t', x_test_padded.shape)
    print()
    print(len(x_train) + len(x_test), 'documents each of length',
          FLAGS.max_doc_length, '.')

    # Store pre-truncated/padded lengths of docs.
    x_len_train = np.array([min(len(x), max_words) for x in x_train])
    x_len_test = np.array([min(len(x), max_words) for x in x_test])
    print('Length of original, unpadded train docs:', x_len_train)
    print('Length of original, unpadded test docs:', x_len_test)

    # Oversample training set to compensate for unbalanced labels.
    # sampler = RandomOverSampler()
    # # Flatten labels array to have shape (n_samples, ) on input.
    # x_train, y_train = sampler.fit_resample(x_train_padded, y_train.flatten())

    # TODO Add data loader to create train/test splits.
    # Turn this into a dataset class.

    return x_train_padded, x_len_train, y_train, x_test_padded, x_len_test, y_test


# Get prepared data.
x_train, x_len_train, y_train, x_test, x_len_test, y_test = prepare_data(data)


# Define input methods for estimator.
def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y


def train_input_fn(params):
    dataset = tf.data.Dataset.from_tensor_slices((x_train,
                                                  x_len_train,
                                                  y_train))
    dataset = dataset.shuffle(buffer_size=120000)
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.map(parser)
    dataset = dataset.repeat(count=FLAGS.num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(params):
    dataset = tf.data.Dataset.from_tensor_slices((x_test,
                                                  x_len_test,
                                                  y_test))
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def predict_input_fn(params):
    dataset = tf.data.Dataset.from_tensor_slices((x_test,
                                                  x_len_test,
                                                  y_test))
    dataset = dataset.batch(batch_size=64)
    dataset = dataset.map(parser)
    # Take out top 10 samples from test data to make the predictions.
    dataset = dataset.take(128)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    inputs = {
        'x': tf.placeholder(dtype=tf.int64,
                            shape=[1, 400],
                            name='input_example_tensor')
    }

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


# Create and run model.
def run_model(features, params, training=True):

    # Word embeddings input layer.
    vocab_size = len(vocab)
    word_embedding_matrix = vocab.astype(np.float32)
    word_embedding_size = word_embedding_matrix.shape[1]
    word_embedding_input_layer = tf.contrib.layers.embed_sequence(
        features['x'],
        vocab_size,
        word_embedding_size,
        scope='words',
        initializer=params['word_embedding_initializer'])

    # Lexicon embeddings need to be created as a numpy array with the indexes
    # matching the indexes of the word embedding indexes.
    # Lex size is the 0th dimension - the number of entries in the lexicon.
    # lex_embedding_size is the 1st dimension, which is the number for dimensions
    # of information about each entry.

    # Lexicon embeddings input layer.
    lex_size = len(lex_emb)
    lex_embedding_matrix = lex_emb.astype(np.float32)
    lex_embedding_size = lex_embedding_matrix.shape[1]
    lex_embedding_input_layer = tf.contrib.layers.embed_sequence(
        features['x'],
        lex_size,
        lex_embedding_size,
        scope='lex',
        initializer=params['lex_embedding_initializer'])

    dropout_word_emb = tf.layers.dropout(inputs=word_embedding_input_layer,
                                         rate=FLAGS.dropout,
                                         training=training)
    dropout_lex_emb = tf.layers.dropout(inputs=lex_embedding_input_layer,
                                        rate=FLAGS.dropout,
                                        training=training)

    # Build word embedding and lexicon attention vectors.

    # Create a convolution + maxpool layer for each filter size
    sequence_length = FLAGS.max_doc_length
    filter_sizes = [3, 4, 5]
    num_filters = 256
    num_filters_lex = 9

    embedding_size = word_embedding_size
    # Expand dimension from end in order to fit 2d conv.
    embedded_chars_expanded = tf.expand_dims(dropout_word_emb, -1)

    embedding_size_lex = lex_embedding_size
    # Expand dimension from end in order to fit 2d conv.
    embedded_chars_expanded_lexicon = tf.expand_dims(dropout_lex_emb, -1)

    h_list = []
    h_lex_list = []
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            U_shape = [num_filters, num_filters_lex]  # (256, 9)
            # These initializers must be created as lambdas to be inside a loop.
            # Ref: https://stackoverflow.com/questions/48515034/keras-tensorflow-initializer-for-variable-is-from-inside-a-control-flow-con?rq=1
            U_trunc_norm = lambda: tf.truncated_normal(U_shape, stddev=0.1)
            U = tf.Variable(U_trunc_norm, name="U")

            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters] # (3, 200, 1, 256)
            W_E_trunc_norm = lambda: tf.truncated_normal(filter_shape, stddev=0.1)
            W_E = tf.Variable(W_E_trunc_norm,
                              name="W_E")
            b_E_const = lambda: tf.constant(0.1, shape=[num_filters])
            b_E = tf.Variable(b_E_const, name="b_E")

            conv_E = tf.nn.conv2d(
                embedded_chars_expanded,
                W_E,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            # Apply nonlinearity
            w2v_conv = tf.nn.relu(tf.nn.bias_add(conv_E, b_E),
                                  name="relu_E")  # (?, 59, 1, 256)
            h_list.append(w2v_conv)

            # for the lex
            filter_shape = [filter_size, embedding_size_lex, 1, num_filters_lex]
            W_L_trunc_norm = lambda: tf.truncated_normal(filter_shape, stddev=0.1)
            W_L = tf.Variable(W_L_trunc_norm,
                              name="W_L")
            b_L_const = lambda: tf.constant(0.1, shape=[num_filters_lex])
            b_L = tf.Variable(b_L_const, name="b_L")

            conv_L = tf.nn.conv2d(
                embedded_chars_expanded_lexicon,
                W_L,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            lex_conv = tf.nn.relu(tf.nn.bias_add(conv_L, b_L),
                                  name="relu_L")  # (?, 59, 1, 9)
            h_lex_list.append(lex_conv)

            w2v_sq = tf.squeeze(w2v_conv, squeeze_dims=[2])  # (?, 59, 256)
            lex_sq = tf.squeeze(lex_conv, squeeze_dims=[2])  # (?, 59, 9)

            print('[w2v_sq]', w2v_sq)

            w2v_sq_tr = tf.matrix_transpose(w2v_sq)
            print('[w2v_sq_tr]', w2v_sq_tr)

            lex_sq_tr = tf.matrix_transpose(lex_sq)
            print('[lex_sq_tr]', lex_sq_tr)

            def fn(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(U, current_input)
                print('previous_output', previous_output)
                print('current_ouput', current_ouput)
                return current_ouput

            initializer = tf.constant(np.zeros([num_filters,
                                                sequence_length - filter_size + 1]),
                                      dtype=tf.float32)

            Ulex = tf.scan(fn, lex_sq_tr, initializer=initializer)
            print('[Ulex]', Ulex)

            WUL = tf.matmul(w2v_sq, Ulex)
            print('[WUL]', WUL)

            WUL_expanded = tf.expand_dims(WUL, -1)
            print('[WUL_expanded]', WUL_expanded)

            # Maxpooling over the outputs
            row_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, 1, sequence_length - filter_size + 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="row_pool")

            print('[row_pool]', row_pool)

            col_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="col_pool")

            print('[col_pool]', col_pool)

            row_pool_sq = tf.expand_dims(tf.squeeze(row_pool, squeeze_dims=[2, 3]),
                                         -1)  # (?, 59, 256)
            print('[row_pool_sq]', row_pool_sq)

            col_pool_sq = tf.expand_dims(tf.squeeze(col_pool, squeeze_dims=[1, 3]),
                                         -1)  # (?, 59, 256)
            print('[col_pool_sq]', col_pool_sq)

            print('[w2v_sq_tr]', w2v_sq_tr)
            print('[lex_sq_tr]', lex_sq_tr)

            attentioned_w2v = tf.matmul(w2v_sq_tr, col_pool_sq)
            attentioned_lex = tf.matmul(lex_sq_tr, row_pool_sq)

            attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            print('[attentioned_w2v]', attentioned_w2v_sq)
            print('[attentioned_lex]', attentioned_lex_sq)

            pooled_outputs.append(attentioned_w2v_sq)
            pooled_outputs.append(attentioned_lex_sq)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(
        filter_sizes)
    print('[pooled_outputs]', len(pooled_outputs))
    h_pool = tf.concat(pooled_outputs, 1)
    print('[self.h_pool]', h_pool)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    print('[self.h_pool_flat]', h_pool_flat)

    doc_embed = tf.nn.dropout(h_pool_flat, FLAGS.dropout)

    # l2 regularization for weights and bias.
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    logits = tf.layers.dense(inputs=doc_embed, units=2,
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer)

    return logits


# Define model.
def cnn_model_fn(features, labels, mode, params):
    # If mode is training, training will be True.
    training = mode == tf.estimator.ModeKeys.TRAIN

    if mode == tf.estimator.ModeKeys.PREDICT:

        # When mode is predict, features will be a single example.
        examples = features

        # Get logits from running model with example.
        logits = run_model(examples, params, training=False)

        # Build predictions dict with probabilities.
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }

        # If running on TPU, use the required TPU-specific stuff.
        if FLAGS.use_tpu:
            # Build and return the estimator spec w/o metrics for training the model.
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                }
            )
        else:
            # Return EstimatorSpec for predictions and outputs defined.
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                }
            )

    # Create model.
    logits = run_model(features, params, training=True)

    # Do not run softmax before passing to softmax_cross_entropy_with_logits_v2.
    # prediction = tf.nn.softmax(output, name="prediction")

    # Transform labels to the form [[1, 0], [0, 1], ...]
    one_hot_labels = tf.one_hot(labels, depth=2)
    one_hot_labels_rs = tf.expand_dims(one_hot_labels, 0)

    # Compute the loss.
    # softmax_cross_entropy_with_logits_v2() computes the softmax and the loss, so
    # the raw logits must be passed into it.
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=one_hot_labels_rs
        )
    )

    # TODO Add l2 loss to final loss here?
    # Ref: https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers
    # l2_loss = tf.losses.get_regularization_loss()
    # loss += l2_loss

    # This will be None when predicting
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    # Create optimizer.
    # TODO Understand the params on this.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, 0.9, 0.99)

    # If running on TPU, use the required TPU-specific stuff.
    if FLAGS.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Evaluation mode.
    if mode == tf.estimator.ModeKeys.EVAL:
        # Define the metrics to compute.
        def metric_fn(labels, logits):
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=tf.argmax(logits, axis=1),
                                           name="accuracy")

            precision = tf.metrics.precision(labels=labels,
                                             predictions=tf.argmax(logits, axis=1),
                                             name="precision")

            recall = tf.metrics.recall(labels=labels,
                                       predictions=tf.argmax(logits, axis=1),
                                       name="recall")

            auc = tf.metrics.auc(labels=labels,
                                 predictions=tf.argmax(logits, axis=1),
                                 name="auc")

            f1 = tf.contrib.metrics.f1_score(labels=labels,
                                             predictions=tf.argmax(logits, axis=1),
                                             name="f1")
            # TODO This is coupled to binary classification, would have to be changed
            # to handle more than two classes for multilabel/multiclass.
            f1_neg = tf.contrib.metrics.f1_score(labels=labels,
                                                 predictions=tf.argmin(logits,
                                                                       axis=1),
                                                 name="f1_neg")

            tp = tf.metrics.true_positives(labels=labels,
                                           predictions=tf.argmax(logits, axis=1),
                                           name="tp")

            tn = tf.metrics.true_negatives(labels=labels,
                                           predictions=tf.argmax(logits, axis=1),
                                           name="tn")

            fp = tf.metrics.false_positives(labels=labels,
                                            predictions=tf.argmax(logits, axis=1),
                                            name="fp")

            fn = tf.metrics.false_negatives(labels=labels,
                                            predictions=tf.argmax(logits, axis=1),
                                            name="fn")

            return {"accuracy": accuracy, "precision": precision,
                    "recall": recall, "auc": auc, "f1": f1,
                    "f1_neg": f1_neg,
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

        # If running on TPU, use the required TPU-specific stuff.
        if FLAGS.use_tpu:
            # Build/return the estimator spec with metrics for evaluating the model.
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=(metric_fn, [labels, logits])
            )
        else:
            # Not on TPU, so use the standard stuff.
            metrics_dict = {
                'accuracy': tf.metrics.accuracy(labels=labels,
                                                predictions=tf.argmax(logits,
                                                                      axis=1),
                                                name="accuracy"),
                'precision': tf.metrics.precision(labels=labels,
                                                  predictions=tf.argmax(logits,
                                                                        axis=1),
                                                  name="precision"),
                'recall': tf.metrics.recall(labels=labels,
                                            predictions=tf.argmax(logits, axis=1),
                                            name="recall"),
                'auc': tf.metrics.auc(labels=labels, predictions=tf.argmax(logits,
                                                                           axis=1),
                                      name="auc"),
                'f1': tf.contrib.metrics.f1_score(labels=labels,
                                                  predictions=tf.argmax(logits,
                                                                        axis=1),
                                                  name="f1"),
                'tp': tf.metrics.true_positives(labels=labels,
                                                predictions=tf.argmax(logits,
                                                                      axis=1),
                                                name="tp"),
                'tn': tf.metrics.true_negatives(labels=labels,
                                                predictions=tf.argmax(logits,
                                                                      axis=1),
                                                name="tn"),
                'fp': tf.metrics.false_positives(labels=labels,
                                                 predictions=tf.argmax(logits,
                                                                       axis=1),
                                                 name="fp"),
                'fn': tf.metrics.false_negatives(labels=labels,
                                                 predictions=tf.argmax(logits,
                                                                       axis=1),
                                                 name="fn")
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics_dict
            )

    # If running on TPU, use the required TPU-specific stuff.
    if FLAGS.use_tpu:
        # Build and return the estimator spec w/o metrics for training the model.
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_global_step())
        )
    else:
        # Not on TPU, so use the standard stuff.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_global_step()))


def main(argv):
    del argv  # Unused.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Initializes the word embeddings for the embedding input in the model.
    def word_embedding_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        # Switch embeddings initialization on flags.
        if FLAGS.random_init_embeddings:
            embedding_matrix = tf.random_uniform_initializer(-1.0, 1.0)
        else:
            embedding_matrix = vocab.astype(np.float32)
        return embedding_matrix

    # Initializes the lexicon embeddings for the embedding input in the model.
    def lex_embedding_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        return lex_emb.astype(np.float32)

    # Assemble hyperparams to pass to estimator.
    params = {'word_embedding_initializer': word_embedding_initializer,
              'lex_embedding_initializer': lex_embedding_initializer}

    # If running on TPU, create the required TPU-specific stuff.
    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project
        )

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
        )

        # Create instance of the TPU estimator for running in all modes.
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=cnn_model_fn,
            model_dir=FLAGS.model_dir,
            use_tpu=FLAGS.use_tpu,
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size,
            predict_batch_size=FLAGS.batch_size,
            params=params,
            config=run_config
        )
    else:
        # TODO Configure session for CPU/GPU.
        session_config = tf.ConfigProto(
            inter_op_parallelism_threads=0,  # Default from MNIST example.
            intra_op_parallelism_threads=0,  # Default from MNIST example.
            allow_soft_placement=True)

        # TODO Configure dist. strategy for CPU/GPU.
        # distribution_strategy = distribution_utils.get_distribution_strategy(
        #     distribution_strategy='default',  # Default from MNIST example.
        #     num_gpus=1 if tf.test.is_gpu_available() else 0,  # Default, MNIST ex.
        #     all_reduce_alg=None)  # Default from MNIST example.

        run_config = tf.estimator.RunConfig(
            # train_distribute=distribution_strategy,  # TODO Not needed?
            session_config=session_config)

        # Add batch size param for cross-compatibility with TPU-compatible use of
        # Data API.
        # TODO Change this to a flag?
        params.update({'batch_size': 16})

        # Create instance of the standard estimator for running in all modes.
        estimator = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=FLAGS.model_dir,
            # train_batch_size=FLAGS.batch_size,
            # eval_batch_size=FLAGS.batch_size,
            # predict_batch_size=1,
            params=params,
            config=run_config
        )

    # Train the model
    estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)

    # Evaluate the model.
    eval_results = estimator.evaluate(input_fn=eval_input_fn,
                                      steps=FLAGS.eval_steps)
    print('\nEvaluation results:\n\t%s\n' % eval_results)

    # TODO Pull this CSV writing stuff out into a function.
    # Build filename of CSV.
    csv_name = Path(
        'results-' + FLAGS.exp_name + '-exp-' + str(FLAGS.exp_num) + '.csv'
    )

    # Build path to which to write CSV.
    csv_path = Path(FLAGS.csv_path / csv_name)

    # Check for existence of csv to determine if header is needed.
    results_file_exists = os.path.isfile(str(csv_path))

    # Open file, write/append results.
    with open(str(csv_path), mode='a') as csv_file:
        # fieldnames = list(eval_results.keys())
        fieldnames = ['global_step', 'loss', 'accuracy', 'precision',
                      'recall', 'auc', 'f1', 'f1_neg', 'tp', 'tn', 'fp', 'fn']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if results csv did not exist at beginning
        # of this trip through.
        if not results_file_exists:
            writer.writeheader()

        # Write row for each run.
        writer.writerow(eval_results)

    # Run prediction on samples of test data.
    if FLAGS.enable_predict:
        predictions = estimator.predict(input_fn=predict_input_fn)

        for pred_dict in predictions:
            template = 'Prediction is "{}" ({:.1f}%).'

            class_id = pred_dict['classes']
            probability = pred_dict['probabilities'][class_id]

            print(template.format(class_id, 100 * probability))

    # Export model to a servable/deployable form.
    if FLAGS.save_model:
        estimator.export_savedmodel(FLAGS.export_dir_base,
                                    serving_input_receiver_fn,
                                    strip_default_attrs=True)

    # Build and send notification message via slack.
    if FLAGS.use_slack:
        text = "\nGlobal step {} for {} #{} done.\nEval results: {}".format(
            eval_results['global_step'],
            FLAGS.exp_name,
            FLAGS.exp_num,
            eval_results)
        message = json.dumps({
            'text': text
        })

        requests.post(FLAGS.slack_webhook_url, message)


if __name__ == "__main__":
    tf.app.run(main)
