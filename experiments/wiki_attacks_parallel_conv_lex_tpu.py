import os
import logging
import sys
import csv
from pathlib import Path
from io import BytesIO

import msgpack
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.lib.io import file_io


print('TensorFlow version:', tf.__version__)

# Cloud TPU Cluster Resolver flags.
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Store CLI args as flags in tf.

# Experiment info.
tf.flags.DEFINE_string("exp_name", "Default-Experiment",
                       "Name of the experiment to which the run belongs.")
tf.flags.DEFINE_integer("exp_num", None, "Number of the experiment to which the "
                                         "run belongs.")

# Paths.
tf.flags.DEFINE_string("data_dir", "",
                       "Path to the data in messagepack bin format.")
tf.flags.DEFINE_string("embeddings_dir", "",
                       "Path to the word embeddings in numpy format.")
tf.flags.DEFINE_string("decoding_dic_path", "",
                       "Path to id2word dictionary in messagepack bin format.")
tf.flags.DEFINE_string("lexicons_path", "lexicons/lex_emb.bin",
                       "Path to lexicons config file.")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_string("csv_path", "", "Path to which to write CSV of results.")

# Batch settings.
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")

# Hyperparameters.
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout rate.")
tf.flags.DEFINE_integer("max_doc_length", 500, "Maximum num of words per document.")
tf.flags.DEFINE_bool("random_init_embeddings", False,
                     "Use random uniform data to initialize word embeddings.")
tf.flags.DEFINE_float("dev_set_size", 0.10,
                      "Size of the development set in percentage of total.")

# Modes.
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training epochs.")

# TPU-specific.
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS

# Read in word embeddings.
logging.info('Loading data.')
f = BytesIO(file_io.read_file_to_string(FLAGS.embeddings_dir, binary_mode=True))
vocab = np.load(f)

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

# Visual check to make sure is2word dict is read in properly.
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
    # TODO The purpose of this has changed - update comments/docs.
    # Convert data to numpy arrays.
    logging.info('Converting data to arrays.')

    # For keeping number of words in longest document in data.
    max_words = 0

    # Create arrays to store docs and labels.
    docs = []
    labels = []

    # Iterate over data to build arrays of docs and labels.
    num_docs = len(raw_data)
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i + 1, num_docs))
        sys.stdout.flush()

        # Get index of document.
        doc = raw_data[i]['idx']

        # Retrieve document from saved data and cast to array.
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

    # TODO Use sklearn to make train/test split?
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
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.map(parser)
    dataset = dataset.repeat(FLAGS.num_epochs)
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


# Define model.
def cnn_model_fn(features, labels, mode, params):

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
    # Lex size would be the 1st dimension, the number of entries in the lexicon.
    # lex_embedding_size would be 2nd dimension.

    # Lexicon embeddings input layer.
    # TODO lex needs to be read in and prepared above.
    lex_size = len(lex_emb)
    lex_embedding_matrix = lex_emb.astype(np.float32)
    lex_embedding_size = lex_embedding_matrix.shape[1]
    lex_embedding_input_layer = tf.contrib.layers.embed_sequence(
        features['x'],
        lex_size,
        lex_embedding_size,
        scope='lex',
        initializer=params['lex_embedding_initializer'])

    # # Lexicon input.
    # input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length,
    #                                               embedding_size_lex],
    #                                  name="input_x_lexicon")
    #
    # # lexicon embeddings.
    # embedded_chars_lexicon = input_x_lexicon
    # embedded_chars_expanded_lexicon = tf.expand_dims(
    #     self.embedded_chars_lexicon, -1)

    training = mode == tf.estimator.ModeKeys.TRAIN

    dropout_word_emb = tf.layers.dropout(inputs=word_embedding_input_layer,
                                         rate=FLAGS.dropout,
                                         training=training)
    dropout_lex_emb = tf.layers.dropout(inputs=lex_embedding_input_layer,
                                        rate=FLAGS.dropout,
                                        training=training)

    # Convolutional layers.
    num_filters = 300
    conv3 = tf.layers.conv1d(
        inputs=dropout_word_emb,
        filters=num_filters,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.reduce_max(input_tensor=conv3, axis=1)

    conv4 = tf.layers.conv1d(
        inputs=dropout_word_emb,
        filters=num_filters,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.reduce_max(input_tensor=conv4, axis=1)

    conv5 = tf.layers.conv1d(
        inputs=dropout_word_emb,
        filters=num_filters,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    pool5 = tf.reduce_max(input_tensor=conv5, axis=1)

    # Lexicon convolutional layers.
    lex_conv3 = tf.layers.conv1d(
        inputs=dropout_lex_emb,
        filters=num_filters,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)
    lex_pool3 = tf.reduce_max(input_tensor=lex_conv3, axis=1)

    lex_conv4 = tf.layers.conv1d(
        inputs=dropout_lex_emb,
        filters=num_filters,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu)
    lex_pool4 = tf.reduce_max(input_tensor=lex_conv4, axis=1)

    lex_conv5 = tf.layers.conv1d(
        inputs=dropout_lex_emb,
        filters=num_filters,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    lex_pool5 = tf.reduce_max(input_tensor=lex_conv5, axis=1)

    # Concat the convolved word embeddings and lex embeddings.
    doc_embed = tf.concat([pool3, pool4, pool5, lex_pool3, lex_pool4, lex_pool5], 1)
    doc_embed = tf.nn.dropout(doc_embed, FLAGS.dropout)

    logits = tf.layers.dense(inputs=doc_embed, units=2)

    # Do not run softmax before passing to softmax_cross_entropy_with_logits_v2.
    # prediction = tf.nn.softmax(output, name="prediction")

    # Transform labels to the form [[0, 0], [0, 1], ...]
    one_hot_labels = tf.one_hot(labels, depth=2)
    one_hot_labels_rs = tf.expand_dims(one_hot_labels, 0)

    # Compute the loss.
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=one_hot_labels_rs
        )
    )

    # dropout_hidden = tf.layers.dropout(inputs=hidden,
    #                                    rate=dropout,
    #                                    training=training)
    #
    # logits = tf.layers.dense(inputs=dropout_hidden, units=1)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # This will be None when predicting
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])

    # Create optimizer.
    # TODO Understand the params on this.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, 0.9, 0.99)

    # Wrap optimizer in TPU-compatible optimizer.
    # TODO Need conditional here to only do this when use_tpu is True.
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
                                             name="F1")

            return {"accuracy": accuracy, "precision": precision,
                    "recall": recall, "auc": auc, "F1": f1}

        # Build and return the estimator spec with metrics for evaluating the model.
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metrics=(metric_fn, [labels, logits])
        )

    # Build and return the estimator spec w/o metrics for training the model.
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_global_step())
    )


def main(argv):
    del argv  # Unused.
    tf.logging.set_verbosity(tf.logging.INFO)

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

    # Create instance of the estimator for running in all modes.
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

    # run_train(args.v, args.t, args.d, args.m, lex_list, args.w2vnumfilters,
    #           args.lexnumfilters, args.randomseed, args.num_epochs,
    #           args.num_class, args.max_sentence_size, args.l2_reg_lambda,
    #           args.l1_reg_lambda, simple_run=False)

    # Train the model.
    estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)

    # Evaluate the model.
    eval_results = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.eval_steps)
    print('\nEvaluation results:\n\t%s\n' % eval_results)

    # TODO Pull this CSV writing stuff out into a function.
    # Build filename of CSV.
    csv_name = Path('results-' + FLAGS.exp_name + '-exp-' + str(FLAGS.exp_num) + '.csv')

    # Build path to which to write CSV.
    csv_path = Path(FLAGS.csv_path / csv_name)

    # Check for existence of csv to determine if header is needed.
    results_file_exists = os.path.isfile(str(csv_path))

    # Open file, write/append results.
    with open(str(csv_path), mode='a') as csv_file:
        # fieldnames = list(eval_results.keys())
        fieldnames = ['global_step', 'loss', 'accuracy', 'precision',
                      'recall', 'auc', 'F1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if results csv did not exist at beginning
        # of this trip through.
        if not results_file_exists:
            writer.writeheader()

        # Write row for each run.
        writer.writerow(eval_results)


if __name__ == "__main__":
    tf.app.run(main)
