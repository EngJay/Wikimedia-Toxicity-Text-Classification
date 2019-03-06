#! /usr/bin/env python

# Wikimedia Toxicity Text Classification Project

import os
import sys
from pathlib import Path
import getopt
import logging
import numpy as np
import tensorflow as tf
import msgpack
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import tf_gao_text_cnn as cnn


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


def main(argv):
    """

    :param argv: array of command line arguments
    :return: None
    """
    # ======================= +
    #                        /
    #    D E F A U L T S    /
    #                      /
    # ------------------- +

    # Default log level.
    logging.basicConfig(level=logging.INFO)

    # Default data directory.
    data_dir = r'data'

    # Default embeddings path.
    embeddings_path = Path('')

    # Default data path.
    data_path = Path('')

    # Label encoding skipped by default
    encode_labels_needed = False

    # Default number of filters.
    # TODO Appropriate default number of filters?
    num_filters = 50

    # Default number of epochs.
    num_epochs = 50

    # Default number of cross-validation splits.
    num_cv_runs = 10
    num_cv_splits = 10

    # Default number of records to use.
    # -1 == use all records.
    num_records = -1

    # ================================= +
    #                                  /
    #    P A R S E  C L I  A R G S    /
    #                                /
    # ----------------------------- +

    # Parse cli args.
    try:
        opts, args = getopt.getopt(argv, "lhve:c:m:n:r:",
                                   ["num_epochs=", "num_cv_splits=",
                                    "embeddings_path=", "data_path=",
                                    "records_num="])
    except getopt.GetoptError:
        print('train.py -e <num_epochs> -c <num_cv_splits> -m <embeddings_path> '
              + '-n <data_path> -r <records_num>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -e <num_epochs> -c <num_cv_splits>\n'
                  + 'Defaults:\n'
                  + '  num_epochs=50\t\tNumber of epochs\n'
                  + '  num_cv_splits=50\tNumber of cross-validation splits\n'
                  + '  embeddings_path=\tPath to word embeddings\n'
                  + '  data_path=\tPath to data\n'
                  + '  records_num=\tNumber of records to pass in\n')
            sys.exit()
        elif opt == '-v':
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt in ("-e", "--num_epochs"):
            num_epochs = int(arg)
        elif opt in ("-c", "--num_cv_splits"):
            num_cv_runs = int(arg)
            num_cv_splits = int(arg)
        elif opt in ("-m", "--embeddings_path"):
            embeddings_path = Path(str(arg))
        elif opt in ("-n", "--data_path"):
            data_path = Path(str(arg))
        elif opt in ("-r", "--records_num"):
            num_records = int(arg)
        elif opt == '-l':
            encode_labels_needed = True

    # TODO Do something with the rest of the hyperparams / params.
    # LABEL_POS_THRESH = 4
    # FEATURE_SIZE = 128
    # NUM_WORDS = 21985
    # EMBED_DIM = 300
    #
    # # TODO IMPROVEMENT Some of these should be cli params w/ default values.
    # # Hyperparameters.
    # # LEARN_RATE = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,]
    # LEARN_RATE = 0.0001
    # BATCH_SIZE = 32
    # DROPOUT = 0.5
    # CONV_DROPOUT = 0.1
    #
    # # Length of longest comment.
    # sequence_length = features.comment.map(len).max()
    #
    # # Number of classes.
    # num_classes = 2
    #
    # # Number of words in vocabulary in embeddings.
    # # vocab_size = len(embed['vocab'])
    #
    # # Number of dimensions in embeddings.
    # embedding_size  = 300
    #
    # # Sizes of filters.
    # filter_sizes = [3, 4, 5]
    #
    # learning_rate = 0.0001
    # batch_size = 32
    # dropout = 0.5
    # conv_dropout = 0.1

    # ======================================== +
    #                                         /
    #    L O A D  &  F O R M A T  D A T A    /
    #                                       /
    # ------------------------------------ +

    # Read in saved files.
    logging.info('Loading data.')
    vocab = np.load(embeddings_path)

    # Load features and labels.
    with open(str(data_path), 'rb') as f:
        data = msgpack.unpack(f, raw=False)

    # Number of docs, which is also the number of observations or samples,
    # and the number of rows of the input matrix.
    if num_records < 0:
        num_docs = len(data)
    else:
        # Subtract 1 to shift to zero-indexing.
        num_docs = num_records - 1

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
    if encode_labels_needed:
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

    # TODO Delete this after duplicating anything beneficial from it w/ CV.
    # Test train split.
    # if encode_labels_needed:
    #     X_train, X_test, y_train, y_test = train_test_split(docs, y_bin,
    #                                                         test_size=0.1,
    #                                                         random_state=1234,
    #                                                         stratify=y)
    # else:
    #     X_train, X_test, y_train, y_test = train_test_split(docs, y_bin,
    #                                                         test_size=0.1,
    #                                                         random_state=1234)

    # Create and train nn.
    logging.info("Building NN.")

    # Cross-validation.
    #   - K-Fold splits K times without considering distribution of classes.
    #   - Stratified K-Fold splits K times while ensuring distribution of classes is
    #     consistent across folds.
    #
    # cv = KFold(CV_SPLITS, True)
    cv = StratifiedKFold(num_cv_splits, True)

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

    # Create instance of the neural network.
    nn = cnn.GaoTextCNN(vocab, num_classes, max_words)

    # Train the NN num_epochs x num_cv_runs.
    #   Default: 10 epochs x 10 CV splits = 100 sessions.
    split_num = 0
    for train, test in cv.split(docs, y):
        split_num += 1
        logging.info('CV Split {0:d}'.format(split_num))

        # TODO Return set of metrics after each epoch.
        nn.train(docs_arr[train], y_arr[train],
                 epochs=num_epochs,
                 cv_split_num=split_num,
                 validation_data=(docs_arr[test], y_arr[test]))

        if split_num == num_cv_runs:
            break

    # ===================== +
    #                      /
    #    M E T R I C S    /
    #                    /
    # ----------------- +

    # TODO Write/report set of metrics after gathering during the training/CV loops?


if __name__ == '__main__':
    main(sys.argv[1:])
