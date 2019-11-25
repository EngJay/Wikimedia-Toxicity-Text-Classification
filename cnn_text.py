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
import requests
import json


print('TensorFlow version:', tf.__version__)

# ======================== +
#                         /
#    S E T  F L A G S    /
#                       /
# -------------------- +

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
# Experiment info.
tf.flags.DEFINE_string("exp_name", "Default-Experiment",
                       "Name of the experiment to which the run belongs.")
tf.flags.DEFINE_integer("exp_num", None,
                        "Number of the experiment to which the run belongs.")

# Paths.
tf.flags.DEFINE_string("data_dir", "",
                       "Path to the data in messagepack binary format.")
tf.flags.DEFINE_string("embeddings_dir", "",
                       "Path to the word embeddings in numpy format.")
tf.flags.DEFINE_string("lexicons_path", "",
                       "Path to lexicons config file.")
tf.flags.DEFINE_string("decoding_dic_path", "",
                       "Path to id2word dictionary in messagepack bin format.")
tf.flags.DEFINE_string("model_dir", None,
                       "Estimator model_dir")
tf.flags.DEFINE_string("csv_path", "",
                       "Path to which to write CSV of results.")

# TPU-specific.
tf.flags.DEFINE_bool("use_tpu", True,
                     "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8,
                        "Number of shards (TPU chips).")

# Export model settings.
tf.flags.DEFINE_bool("save_model", False,
                     "Export trained estimator.")
tf.flags.DEFINE_string("export_dir_base", None,
                       "Base path to which exported model will be saved.")

# Hyperparameters for model and modes.
tf.flags.DEFINE_float("learning_rate", 0.001,
                      "Learning rate.")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_float("dropout", 0.5,
                      "Dropout rate.")
tf.flags.DEFINE_integer("max_doc_length", 400,
                        "Maximum num of words per document.")
tf.flags.DEFINE_integer("train_steps", 1000,
                        "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("dev_set_size", 0.10,
                      "Size of the development set as fraction of total.")
tf.flags.DEFINE_integer("num_epochs", None,
                        "Number of training epochs.")
tf.flags.DEFINE_bool("random_init_embeddings", False,
                     "Use random uniform data to initialize embeddings.")
tf.flags.DEFINE_integer("num_classes", 2,
                        "Number of classes.")

# Modes.
tf.flags.DEFINE_bool("train_enabled", False,
                     "Run the estimator in training mode.")
# TODO Refactor to decouple eval from training.
tf.flags.DEFINE_bool("eval_enabled", False,
                     "Run the estimator in evaluation mode.")
tf.flags.DEFINE_bool("predict_enabled", False,
                     "Run predictions at the end of training run.")

# Notifications.
tf.flags.DEFINE_bool("use_slack", False,
                     "Use slack to receive notifications when run is done.")
tf.flags.DEFINE_string("slack_webhook_url", None,
                       "Slack app webhook.")

FLAGS = tf.flags.FLAGS

# ======================== +
#                         /
#    R E A D  D A T A    /
#                       /
# -------------------- +

# Read in embeddings.
logging.info('Loading data.')
f = BytesIO(file_io.read_file_to_string(FLAGS.embeddings_dir, binary_mode=True))
vocab = np.load(f)

# TODO Go through embeddings and ensure that the additional random row for unknown
# words is added during preprocessing.
# Add one additional row to embeddings for unknown words.
# Add additional word embedding for unknown words.
vocab = np.concatenate((vocab, np.random.rand(1, vocab.shape[1])))

# Print size of vocab to confirm size.
print('len(vocab):', len(vocab))

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
    # TODO Move this into the TF dataset API?
    # Convert data to numpy arrays.
    logging.info('Converting data to arrays.')

    # For keeping number of words in longest document.
    max_words = 0

    # Create lists to store docs, labels, and rev_ids.
    docs = []
    labels = []
    rev_ids = []

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

        # TODO Pull conditionals for binary vs multilabel out?
        if FLAGS.num_classes > 2:
            # Add labels list to label array at same index.
            casted_doc_labels = list(map(int, raw_data[i]['labels']))
            labels.append(casted_doc_labels)
        else:
            # Add label to label array at same index.
            labels.append(raw_data[i]['label'])

        # Get rev_id of document and add to list.
        rev_id = raw_data[i]['rev_id']
        rev_ids.append(rev_id)

        # Track maximum number of words in document.
        if len(doc) > max_words:
            max_words = len(doc)

    del raw_data
    print()

    # TODO Pull conditionals for binary vs multilabel out?
    if FLAGS.num_classes > 2:
        # TODO Refactor to clean this up.
        y = labels
    else:
        # TODO Still needed?
        # Label encoder.
        #   Encode labels with value between 0 and n_classes-1,
        #   so for example 1 to 5 star ratings become 0 to 4.
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # TODO Still needed?
        # Binarize labels.
        lb = LabelBinarizer()
        y = lb.fit_transform(y)

    # Combine docs and rev_ids in dict, so splitting doesn't mix them up.
    docs_with_rev_ids = []
    for idx, doc in enumerate(docs):
        doc_dict = {'doc': doc, 'rev_id': rev_ids[idx]}
        docs_with_rev_ids.append(doc_dict)

    # Random state is seeded in order to run predictions on the same examples
    # from experiment to experiment.
    # TODO Make random state a flag?
    x_train, x_test, y_train, y_test = train_test_split(docs_with_rev_ids, y,
                                                        test_size=FLAGS.dev_set_size,
                                                        random_state=42)

    # Decode and print first train example.
    print('Decoded first training doc:', indexes_to_text(x_train[0]['doc']))

    # Get just the docs in a list to pass into Kera's pad_sequences().
    x_train_sequences = []
    for idx, doc in enumerate(x_train):
        x_train_sequences.append(doc['doc'])

    x_test_sequences = []
    for idx, doc in enumerate(x_test):
        x_test_sequences.append(doc['doc'])

    # Pad/truncate all docs to max_doc_length using the pad_id char.
    pad_id = 0
    x_train_sequences_padded = sequence.pad_sequences(x_train_sequences,
                                                      maxlen=FLAGS.max_doc_length,
                                                      truncating='post',
                                                      padding='post',
                                                      value=pad_id)

    x_test_sequences_padded = sequence.pad_sequences(x_test_sequences,
                                                     maxlen=FLAGS.max_doc_length,
                                                     truncating='post',
                                                     padding='post',
                                                     value=pad_id)

    # Visual sanity checks.
    print('Unpadded length of first training doc:\t', len(x_train_sequences[0]))
    print('Unpadded length of second training doc:\t', len(x_train_sequences[1]))
    print('Padded len of first training doc:\t', len(x_train_sequences_padded[0]))
    print('Padded len of second training doc:\t', len(x_train_sequences_padded[1]))
    print('x_train_sequences_padded shape:\t\t', x_train_sequences_padded.shape)
    print('x_test_sequences_padded shape:\t\t', x_test_sequences_padded.shape)
    print()
    print(len(x_train) + len(x_test), 'documents each of length',
          FLAGS.max_doc_length, '.')

    # Build lists of rev_ids.
    x_train_rev_ids = []
    for idx, doc in enumerate(x_train):
        x_train_rev_ids.append(int(x_train[idx]['rev_id']))

    x_test_rev_ids = []
    for idx, doc in enumerate(x_test):
        x_test_rev_ids.append(int(x_test[idx]['rev_id']))

    # TODO Add data loader to create train/test splits.
    # Turn this into a dataset class.

    return x_train_sequences_padded, x_train_rev_ids, y_train, \
        x_test_sequences_padded, x_test_rev_ids, y_test


# Get prepared data.
x_train_seqs, x_train_ids, y_train, \
    x_test_seqs, x_test_ids, y_test = prepare_data(data)


# Define input methods for estimator.
def parser(x, rev_id, y):
    features = {'x': x, 'rev_id': rev_id}
    return features, y


def train_input_fn(params):
    dataset = tf.data.Dataset.from_tensor_slices((x_train_seqs,
                                                  x_train_ids,
                                                  y_train))
    dataset = dataset.shuffle(buffer_size=120000)
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.map(parser)
    dataset = dataset.repeat(count=FLAGS.num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(params):
    dataset = tf.data.Dataset.from_tensor_slices((x_test_seqs,
                                                  x_test_ids,
                                                  y_test))
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def predict_input_fn(params):
    dataset = tf.data.Dataset.from_tensor_slices((x_test_seqs,
                                                  x_test_ids,
                                                  y_test))
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.map(parser)
    # Run predictions for the entire set once.
    dataset = dataset.repeat(count=1)
    # Take out top 10 samples from test data to make the predictions.
    # dataset = dataset.take(16)
    # iterator = dataset.make_one_shot_iterator()
    return dataset


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    inputs = {
        'x': tf.placeholder(dtype=tf.int64,
                            shape=[1, 400],
                            name='input_example_tensor')
    }

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def run_model(features, params, training=True):
    """Create and run model."""

    # Word embeddings.
    # Tf embedding layer expects 32-bit float as data type.
    embedding_matrix = vocab.astype(np.float32)
    # Embeddings size is the num of dimensions, or the num of columns of the matrix.
    embedding_size = embedding_matrix.shape[1]
    input_layer = tf.contrib.layers.embed_sequence(
        features['x'],
        len(vocab),
        embedding_size,
        scope='words',
        initializer=params['embedding_initializer'])

    # Place a dropout layer between the embedding input and the convolutional layers.
    dropout_emb = tf.layers.dropout(inputs=input_layer,
                                    rate=FLAGS.dropout,
                                    training=training)

    # Convolutional layers.
    num_filters = 300
    conv3 = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=num_filters,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.reduce_max(input_tensor=conv3, axis=1)

    conv4 = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=num_filters,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.reduce_max(input_tensor=conv4, axis=1)

    conv5 = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=num_filters,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    pool5 = tf.reduce_max(input_tensor=conv5, axis=1)

    # Concatenate the output of the three convolutional layers.
    # This is a parallel approach to a CNN, could also be in series.
    doc_embed = tf.concat([pool3, pool4, pool5], 1)

    # Insert another dropout layer between the concatenated outputs
    # of the the convolutional layers.
    doc_embed = tf.nn.dropout(doc_embed, FLAGS.dropout)

    # Final dense layer with n units to determine logits for the n classes.
    logits = tf.layers.dense(inputs=doc_embed, units=FLAGS.num_classes)

    return logits


def cnn_model_fn(features, labels, mode, params):
    """Define model function for TF estimator."""

    # If mode is training, training will be True.
    training = mode == tf.estimator.ModeKeys.TRAIN

    # Covert predictions to multi-hot for comparison to labels.
    def multi_label_one_hot(prediction, threshold=0.5):
        prediction = tf.cast(prediction, tf.float32)
        threshold = float(threshold)
        return tf.cast(tf.greater(prediction, threshold), tf.int64)

    if mode == tf.estimator.ModeKeys.PREDICT:

        # When mode is predict, features will be a single example.
        example = features

        # Get logits from running model with example.
        logits = run_model(example, params, training=False)

        if FLAGS.num_classes > 2:
            predictions = tf.sigmoid(logits)
            one_hot_predictions = multi_label_one_hot(predictions)
            predictions = {
                'rev_id': features['rev_id'],
                'classes': one_hot_predictions,
                'probabilities': predictions,
            }
        else:
            # Build predictions dict with probabilities.
            predictions = {
                'rev_id': features['rev_id'],
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits)

            }

        # Return EstimatorSpec for predictions and outputs defined.
        if FLAGS.use_tpu:
            # Build/return the estimator spec with metrics for evaluating the model.
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={
                    'classify': tf.estimator.export.PredictOutput(predictions)
                })

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    # Create model.
    logits = run_model(features, params, training=True)

    if FLAGS.num_classes > 2:
        # Compute the loss for multiple labels.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.cast(labels,
                                                                      tf.float32))
        loss = tf.reduce_mean(
            tf.reduce_sum(loss, axis=1)
        )

        predictions = tf.sigmoid(logits)
        one_hot_predictions = multi_label_one_hot(predictions)
    else:
        # Transform labels to the form [[1, 0], [0, 1], ...]
        one_hot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
        # one_hot_labels_rs = tf.expand_dims(one_hot_labels, 0)

        # Compute the loss for binary label.
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=one_hot_labels
            )
        )

        # Get predicted label from logits.
        predictions = tf.argmax(logits, axis=1)

    # Create optimizer for CPU/GPU.
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    # If running on TPU, use the required TPU-specific stuff instead.
    if FLAGS.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    # Evaluation mode.
    if mode == tf.estimator.ModeKeys.EVAL:
        # If running on TPU, use the required TPU-specific stuff.
        if FLAGS.use_tpu:
            # Define the metrics to compute.
            def metric_fn(labels, predictions):
                accuracy = tf.metrics.accuracy(labels=labels,
                                               predictions=predictions,
                                               name="accuracy")

                precision = tf.metrics.precision(labels=labels,
                                                 predictions=predictions,
                                                 name="precision")

                recall = tf.metrics.recall(labels=labels,
                                           predictions=predictions,
                                           name="recall")

                auc = tf.metrics.auc(labels=labels,
                                     predictions=predictions,
                                     name="auc")

                f1 = tf.contrib.metrics.f1_score(labels=labels,
                                                 predictions=predictions,
                                                 name="f1")

                tp = tf.metrics.true_positives(labels=labels,
                                               predictions=predictions,
                                               name="tp")

                tn = tf.metrics.true_negatives(labels=labels,
                                               predictions=predictions,
                                               name="tn")

                fp = tf.metrics.false_positives(labels=labels,
                                                predictions=predictions,
                                                name="fp")

                fn = tf.metrics.false_negatives(labels=labels,
                                                predictions=predictions,
                                                name="fn")

                return {"accuracy": accuracy, "precision": precision,
                        "recall": recall, "auc": auc, "f1": f1,
                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

            # Build/return the estimator spec with metrics for evaluating the model.
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=(metric_fn, [labels, predictions]))
        else:
            # Not on TPU, so use the standard stuff.
            metrics_dict = {
                'accuracy': tf.metrics.accuracy(labels=labels,
                                                predictions=predictions,
                                                name="accuracy"),
                'precision': tf.metrics.precision(labels=labels,
                                                  predictions=predictions,
                                                  name="precision"),
                'recall': tf.metrics.recall(labels=labels,
                                                  predictions=predictions,
                                            name="recall"),
                'auc': tf.metrics.auc(labels=labels,
                                                  predictions=predictions,
                                      name="auc"),
                'f1': tf.contrib.metrics.f1_score(labels=labels,
                                                  predictions=predictions,
                                                  name="f1"),
                'tp': tf.metrics.true_positives(labels=labels,
                                                  predictions=predictions,
                                                name="tp"),
                'tn': tf.metrics.true_negatives(labels=labels,
                                                  predictions=predictions,
                                                name="tn"),
                'fp': tf.metrics.false_positives(labels=labels,
                                                  predictions=predictions,
                                                 name="fp"),
                'fn': tf.metrics.false_negatives(labels=labels,
                                                  predictions=predictions,
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

    # Initializes the word embeddings for the embedding input in the model function.
    def embedding_initializer(shape=None, dtype=tf.float32, partition_info=None):
        assert dtype is tf.float32
        # Switch embeddings initialization on flags.
        if FLAGS.random_init_embeddings:
            embedding_matrix = tf.random_uniform_initializer(-1.0, 1.0)
        else:
            embedding_matrix = vocab.astype(np.float32)
        return embedding_matrix

    # Assemble hyperparams to pass to estimator.
    params = {'embedding_initializer': embedding_initializer}

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
        params.update({'batch_size': FLAGS.batch_size})

        # Create instance of the standard estimator for running in all modes.
        estimator = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=FLAGS.model_dir,
            params=params,
            config=run_config
        )

    if FLAGS.train_enabled:
        estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)

    if FLAGS.eval_enabled:
        # Evaluate the model.
        # TODO Can estimator run just in evaluate mode?
        eval_results = estimator.evaluate(input_fn=eval_input_fn,
                                          steps=FLAGS.eval_steps)
        print('\nEvaluation results:\n\t%s\n' % eval_results)

        # ======================================+
        #                                      /
        #    C O M P U T E  F 1  M A C R O    /
        #                                    /
        # ----------------------------------+

        # Compute f1 macro.
        # TODO Pull this out into a function.
        # TODO Handle n number of labels?
        # For positive label.
        tp_1 = eval_results['tp']
        tn_1 = eval_results['tn']
        fp_1 = eval_results['fp']
        fn_1 = eval_results['fn']

        # Flip values around to treat negative as second label.
        tp_0, tn_0, fp_0, fn_0 = tn_1, tp_1, fn_1, fp_1

        prec_1 = tp_1 / (tp_1 + fp_1)
        recall_1 = tp_1 / (tp_1 + fn_1)
        f1_1 = 2 * ((prec_1 * recall_1) / (prec_1 + recall_1))

        prec_0 = tp_0 / (tp_0 + fp_0)
        recall_0 = tp_0 / (tp_0 + fn_0)
        f1_0 = 2 * ((prec_0 * recall_0) / (prec_0 + recall_0))

        eval_results['f1_macro'] = (f1_1 + f1_0) / 2.0

        # =========================================+
        #                                         /
        #    W R I T E  R E S U L T S  C S V     /
        #                                       /
        # -------------------------------------+

        # TODO Pull this CSV writing stuff out into a function.
        # Build filename of CSV.
        csv_name = Path('results-' + FLAGS.exp_name
                        + '-exp-' + str(FLAGS.exp_num) + '.csv')

        # Build path to which to write CSV.
        csv_path = Path(FLAGS.csv_path / csv_name)

        # Check for existence of csv to determine if header is needed.
        results_file_exists = os.path.isfile(str(csv_path))

        # Open file, write/append results.
        with open(str(csv_path), mode='a') as csv_file:
            fieldnames = ['global_step', 'loss', 'accuracy', 'precision',
                          'recall', 'auc', 'f1', 'f1_macro',
                          'tp', 'tn', 'fp', 'fn']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write header only if results csv did not exist at beginning
            # of this trip through.
            if not results_file_exists:
                writer.writeheader()

            # Write row for each run.
            writer.writerow(eval_results)

    # =====================================+
    #                                     /
    #    R U N  P R E D I C T I O N S    /
    #                                   /
    # ---------------------------------+

    # TODO Add logging of predictions.
    if FLAGS.predict_enabled:
        # Run the predictions.
        predictions = estimator.predict(input_fn=predict_input_fn)

        # Build lists of predicted labels.
        preds_positive = []
        preds_negative = []
        for pred_dict in predictions:
            # print(pred_dict)

            template = 'Prediction of example rev_id {} is "{}" ({:.1f}%).'

            # Cast to native python datatypes.
            rev_id = int(pred_dict['rev_id'])
            class_id = int(pred_dict['classes'])
            probability = float(pred_dict['probabilities'][class_id])

            print(template.format(rev_id, class_id, 100 * probability))

            item = [rev_id, probability]
            if class_id == 1:
                preds_positive.append(item)
            else:
                preds_negative.append(item)

        # Combine lists into dict of predictions.
        pred_dict = {'pos': preds_positive, 'neg': preds_negative}

        print('preds_positive:', preds_positive)
        print('preds_negative:', preds_negative)

        # Build filename of predictions.
        preds_file_name = Path('preds-' + FLAGS.exp_name
                             + '-exp-' + str(FLAGS.exp_num) + '.bin')

        # Build path to which to write predictions dict.
        pred_path = Path(FLAGS.csv_path / preds_file_name)

        # Write dict to file.
        data_output_path = pred_path
        with open(str(data_output_path), 'wb') as f:
            msgpack.pack(pred_dict, f)

    # ===============================+
    #                               /
    #    E X P O R T  M O D E L    /
    #                             /
    # ---------------------------+

    if FLAGS.save_model:
        estimator.export_savedmodel(FLAGS.export_dir_base,
                                    serving_input_receiver_fn,
                                    strip_default_attrs=True)

    # ==================================+
    #                                  /
    #    N O T I F I C A T I O N S    /
    #                                /
    # ------------------------------+

    if FLAGS.use_slack:
        text = "\nExperiment {} #{} done.".format(
            FLAGS.exp_name,
            FLAGS.exp_num)

        if FLAGS.eval_enabled:
            text = "\nGlobal step {} for {} #{} done.\nEval results: {}".format(
                eval_results['global_step'],
                FLAGS.exp_name,
                FLAGS.exp_num,
                eval_results)

        message = json.dumps({
            'text': text
        })

        requests.post(FLAGS.slack_webhook_url, message)

        print()
        print(' # ======================================== +')
        print(' #                                         /')
        print(' #    S E N T  N O T I F I C A T I O N    /')
        print(' #                                       /')
        print(' # ------------------------------------ +')
        print()


if __name__ == "__main__":
    tf.app.run(main)
