#! /usr/bin/env python

# Wikimedia Toxicity Text Classification Project

import os
import sys
import logging
from io import BytesIO
from tensorflow.python.lib.io import file_io
import numpy as np
import tensorflow as tf
import msgpack
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from deprecated import tf_gao_text_cnn as cnn

# GPU check.
# Ensure tensorflow can find CUDA device.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # TODO Determine if this affects GPU use.
# Fix for weird MacOS bug.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Check for GPU connectivity.
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU not found.')
else:
    print('Found GPU at : {}'.format(device_name))

# ================================= +
#                                  /
#    P A R S E  C L I  A R G S    /
#                                /
# ----------------------------- +

# Cloud TPU Cluster Resolver flags
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

# Model specific parameters
# TODO Some of these might not be needed or need modifying.
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing data")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

tf.flags.DEFINE_string('embeddings_path', '', 'Path to word embeddings.')
tf.flags.DEFINE_string('data_path', '', 'Path to data.')
tf.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
tf.flags.DEFINE_integer('num_cv_runs', 10, 'Number of CV splits left to run.')
tf.flags.DEFINE_integer('num_cv_splits', 10, 'Number of CV splits.')
tf.flags.DEFINE_integer('num_records', -1, 'Number of records to run, default'
                                           'of -1 runs all records.')
tf.flags.DEFINE_bool("encode_labels", False, "Encode labels before feeding.")
tf.flags.DEFINE_bool("verbose", False, "Set verbosity to debug.")

FLAGS = tf.flags.FLAGS


def main(argv):
    del argv
    # Set verbosity of tf logging.
    if FLAGS.verbose:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.use_tpu:
        # Configure GC Cluster Resolver for TPU usage.
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project
        )

        # Configure TPU.
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
        )

    # ======================= +
    #                        /
    #    D E F A U L T S    /
    #                      /
    # ------------------- +

    # Default log level.
    logging.basicConfig(level=logging.INFO)

    # Default number of filters.
    # TODO Appropriate default number of filters?
    num_filters = 50

    # ======================================== +
    #                                         /
    #    L O A D  &  F O R M A T  D A T A    /
    #                                       /
    # ------------------------------------ +

    # Read in saved files.
    logging.info('Loading data.')
    f = BytesIO(file_io.read_file_to_string(FLAGS.embeddings_path, binary_mode=True))
    vocab = np.load(f)

    # Load features and labels.
    f = BytesIO(file_io.read_file_to_string(FLAGS.data_path, binary_mode=True))
    data = msgpack.unpack(f, raw=False)

    # Number of docs, which is also the number of observations or samples,
    # and the number of rows of the input matrix.
    if FLAGS.num_records < 0:
        num_docs = len(data)
    else:
        # Subtract 1 to shift to zero-indexing.
        num_docs = FLAGS.num_records - 1

    # Convert data to numpy arrays.
    logging.info('Converting data to arrays.')

    # For keeping number of words in longest document in data.
    max_words = 0

    # Create arrays to store docs and labels.
    docs = []
    labels = []

    # Iterate over data to build arrays of docs and labels.
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i + 1, num_docs))
        sys.stdout.flush()

        # Get index of document.
        doc = data[i]['idx']

        # Retrieve document from saved data and cast to array.
        doc = [item for sublist in doc for item in sublist]

        # Add document to docs array.
        docs.append(doc)

        # Add label to label array at same index.
        labels.append(data[i]['label'])

        # Track maximum number of words in document.
        if len(doc) > max_words:
            max_words = len(doc)

    del data
    print()

    # TODO Better handling of label encoding?
    #   - Develop set of cases to handle and document.
    if FLAGS.encode_labels:
        # Label encoder.
        #   Encode labels with value between 0 and n_classes-1,
        #   so for example 1 to 5 star ratings become 0 to 4.
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Number of classes.
        num_classes = len(le.classes_)

        # One-Hot encode and reshape if less than 3 classes to avoid shape mismatch.
        #  - This is necessary because LabelBinarizer will put the labels into
        #    the shape (,1) when passed data with binary labels, whereas the logic
        #    of Gao's network requires even binary labels to be in the form
        #    (,2), just as One-Hot even though the labels are binary.
        if num_classes < 3:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(y.reshape(-1, 1))
            y_bin = enc.transform(y.reshape(-1, 1)).toarray()
        else:
            # Binarize labels in one-vs-all fashion if three or more classes.
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y)

        del labels
    else:
        # Get number of classes from existing encoded labels.
        num_classes = len(max(labels, key=len))

        # For sake of consistency, use same name for existing encoded labels
        y_bin = labels
        y = labels

    # ====================================== +
    #                                       /
    #    C R O S S  V A L I D A T I O N    /
    #                                     /
    # ---------------------------------- +

    # Create and train nn.
    logging.info("Building NN.")

    # Cross-validation.
    #   - K-Fold splits K times without considering distribution of classes.
    #   - Stratified K-Fold splits K times while ensuring distribution of classes is
    #     consistent across folds.
    #
    # cv = KFold(CV_SPLITS, True)
    cv = StratifiedKFold(FLAGS.num_cv_splits, True)

    # In order to select by set of indices, convert to numpy array.
    docs_arr = np.array(docs)
    y_arr = np.array(y_bin)
    # if num_records >= 0:
    #     docs_arr = docs_arr[1000:]
    #     y_arr = y_arr[1000:]

    # ================================ +
    #                                 /
    #    T R A I N  N E T W O R K    /
    #                               /
    # ---------------------------- +

    logging.info("Training NN.")

    # Train the NN num_epochs x num_cv_runs.
    #   Default: 10 epochs x 10 CV splits = 100 sessions.
    split_num = 0
    for train, test in cv.split(docs, y):
        split_num += 1
        logging.info('CV Split {0:d}'.format(split_num))

        # Create instance of the neural network.
        # TODO Should this be out of the CV loop?
        nn = cnn.GaoTextCNN(vocab, num_classes, max_words, flags=FLAGS)

        # TODO Return set of metrics after each epoch.
        nn.train(docs_arr[train], y_arr[train],
                 epochs=FLAGS.num_epochs,
                 cv_split_num=split_num,
                 validation_data=(docs_arr[test], y_arr[test]))

        if split_num == FLAGS.num_cv_runs:
            break

    # ===================== +
    #                      /
    #    M E T R I C S    /
    #                    /
    # ----------------- +

    # TODO Write/report set of metrics after gathering during the training/CV loops?


if __name__ == '__main__':
    main(sys.argv[1:])
