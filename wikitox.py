#! /usr/bin/env python

# Wikimedia Toxicity Text Classification Project

import os
import sys
from pathlib import Path
import getopt
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import msgpack
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import extract_features as ef
import tf_gao_text_cnn as cnn


# GPU check.
# Ensure tensorflow can find CUDA device.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    # Data preparation skipped by default.
    data_prep_needed = False

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

    # Default number of votes to decide label based on the annotations of the
    # ten workers who annotated the dataset.
    min_num_votes = 6

    # ================================= +
    #                                  /
    #    P A R S E  C L I  A R G S    /
    #                                /
    # ----------------------------- +

    # Parse cli args.
    try:
        opts, args = getopt.getopt(argv, "dhve:c:m:n:",
                                   ["num_epochs=", "num_cv_splits=",
                                    "embeddings_path=", "data_path="])
    except getopt.GetoptError:
        print('train.py -e <num_epochs> -c <num_cv_splits> -m <embeddings_path> '
              + '-n <data_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -e <num_epochs> -c <num_cv_splits>\n'
                  + 'Defaults:\n'
                  + '  num_epochs=50\t\tNumber of epochs\n'
                  + '  num_cv_splits=50\tNumber of cross-validation splits\n'
                  + '  embeddings_path=\tPath to word embeddings\n'
                  + '  data_path=\tPath to data\n')
            sys.exit()
        elif opt == '-v':
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == '-d':
            data_prep_needed = True
        elif opt in ("-e", "--num_epochs"):
            num_epochs = int(arg)
        elif opt in ("-c", "--num_cv_splits"):
            num_cv_runs = int(arg)
            num_cv_splits = int(arg)
        elif opt in ("-m", "--embeddings_path"):
            embeddings_path = Path(str(arg))
        elif opt in ("-n", "--data_path"):
            data_path = Path(str(arg))
        elif opt == '-l':
            encode_labels_needed = True

    # Set vars with default or passed-in values.

    # Skip data preparation by default.
    # TODO Move all of this into a dataset-specific data_prep_DATASET script.
    if data_prep_needed:

        # Get the data, create dataframes from the tab-separated files.
        # TODO Add setting for separator, so not coupled to tsv.
        attacks_comments_path = data_path / 'Wikimedia-Toxicity-Personal-Attacks'
        attacks_comments_path = attacks_comments_path / 'attack_annotated_comments.tsv'
        attacks_comments_df = pd.read_csv(attacks_comments_path, sep='\t', header=0)

        attacks_labels_path = data_path / 'Wikimedia-Toxicity-Personal-Attacks'
        attacks_labels_path = attacks_labels_path / 'attack_annotations.tsv'
        attacks_labels_df = pd.read_csv(attacks_labels_path, sep='\t', header=0)

        logging.debug(attacks_comments_df.head())
        logging.debug(attacks_labels_df.head())

        # Build paths to word embeddings.
        # TODO IMPROVEMENT Make this a cli arg w/ Google vectors as default.
        raw_embeddings_filename = data_path / 'pretrained_embeddings' / 'GoogleNews-vectors-negative300.bin'
        # TODO IMPROVEMENT Create cache file name based on original vector binary.
        embeddings_cache_filename = data_path / 'pretrained_embeddings' / 'embeddings-cache.bin'

        # ============================== +
        #                               /
        #    P R E P A R E  D A T A    /
        #                             /
        # --------------------------- +

        # Merge data frames of comments and annotations on rev_id.
        attacks_merged = pd.merge(attacks_comments_df, attacks_labels_df,
                                  on='rev_id')

        # Treat the 10 records (one for each worker) for each comment
        # like votes: > 5 workers reporting a comment contains an attack = 1.
        # Group by rev_id, then sum attack column per group.
        #
        # Since the presence of an attack is a 1, the annotations
        # by the workers can be treated as votes, so a sum of the
        # attack column greater than 5 means more than half of the
        # workers thought the comment contained a personal attack,
        # and is therefore labeled as containing a personal attack.
        attacks_merged_summed = attacks_merged.groupby('rev_id').sum()

        # LABELS: Build set of rev_ids that contain personal attacks as labels.
        attacks = attacks_merged_summed.loc[attacks_merged_summed['attack'] > 5].copy()
        attacks.reset_index(level=0, inplace=True)
        attacks['attack'] = 1
        attacks.drop(
            ['year', 'logged_in', 'worker_id', 'quoting_attack', 'recipient_attack',
             'third_party_attack', 'other_attack'], axis=1, inplace=True)

        # Build set of rev_ids that do not contain attacks.
        no_attacks = attacks_merged_summed.loc[
            attacks_merged_summed['attack'] <= 5].copy()
        no_attacks.reset_index(level=0, inplace=True)
        no_attacks['attack'] = 0
        no_attacks.drop(
            ['year', 'logged_in', 'worker_id', 'quoting_attack', 'recipient_attack',
             'third_party_attack', 'other_attack'], axis=1, inplace=True)

        # Combine the the two sets and sort.
        labels = attacks.append(no_attacks)
        labels.sort_values(by=['rev_id'], inplace=True)
        labels.reset_index(level=0, drop=True, inplace=True)

        logging.debug(print(labels.head()))

        # FEATURES: Create features.
        # groupby the rev_id, get only first of each group.
        features = attacks_merged.groupby('rev_id').first().copy()

        # Reset index, saving rev_id as column.
        features.reset_index(level=0, inplace=True)

        # Drop everything except for 'rev_id' and 'comment'.
        features.drop(['year', 'logged_in', 'ns', 'sample', 'split', 'worker_id',
                       'quoting_attack', 'recipient_attack', 'third_party_attack',
                       'other_attack', 'attack'], axis=1, inplace=True)

        # Merge with labels for complete set labeled data.
        features = pd.merge(features, labels, on='rev_id').copy()

        # Number of comments with and without attacks.
        num_attacks = len(features[features['attack'] == 1].index)
        num_not_attacks = len(features[features['attack'] == 0].index)

        logging.info(
            print('Num of comments containing an attack: ', num_attacks)
        )
        logging.info(
            print('Num of comments not containing an attack: ', num_not_attacks)
        )

        # Write features and labels to disk.
        # TODO Add setting for path to store cached data.
        raw_features_path = Path('data/cache/raw_features.csv')
        features.to_csv(raw_features_path)

        logging.debug(print(features.head()))

        # =================================================== +
        #                                                    /
        #    P R E P A R E  W O R D  E M B E D D I N G S    /
        #                                                  /
        # ----------------------------------------------- +

        # Build vocabulary and word embeddings from source if needed.

        ef.extract_features(raw_features_path)

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

    # ============================================= +
    #                                              /
    #    B U I L D  N E U R A L  N E T W O R K    /
    #                                            /
    # ----------------------------------------- +

    # Read in saved files.
    print("loading data")
    # TODO Add cached embeddings path to cli args.
    # embeddings_path = Path(r'data') / 'cache' / 'wikimedia-personal-attacks' / \
    #     'wikimedia-personal-attacks-embeddings.npy'
    # embeddings_path = \
    #    Path(r'data') / 'cache' / 'course-reviews-embeddings.npy'
    vocab = np.load(embeddings_path)

    # TODO Add data path to cli args.
    # data_path = Path(r'data') / 'cache' / 'wikimedia-personal-attacks' / \
    #     'wikimedia-personal-attacks-data.bin'
    # data_path = Path(r'data') / 'course_reviews' / 'course-reviews-data.bin'
    with open(str(data_path), 'rb') as f:
        data = msgpack.unpack(f, raw=False)

    # Number of docs, which is also the number of observations or samples,
    # and the number of rows of the input matrix.
    num_docs = len(data)

    # Convert data to numpy arrays.
    print("converting data to arrays")
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

        # Track maximum number of word in document.
        if len(doc) > max_words:
            max_words = len(doc)

    # Delete read-in data from memory.
    del data
    print()

    if encode_labels_needed:
        # Label encoder.
        # Encode labels with value between 0 and n_classes-1,
        # so for example 1 to 5 star ratings become 0 to 4.
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # Number of classes.
        num_classes = len(le.classes_)

        # One-Hot encode if less than 3 classes to avoid
        # tensor shape mismatch.
        if num_classes < 3:
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(y.reshape(-1, 1))
            y_bin = enc.transform(y.reshape(-1, 1)).toarray()
        else:
            # Binarize labels in a one-vs-all fashion if three or
            # more classes.
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y)

        del labels
    else:
        # Get number of classes from existing encoded labels.
        num_classes = len(max(labels, key=len))

        # For sake of consistency, use same name for existing encoded labels
        y_bin = labels
        y = labels

    # TODO Add crossvalidation.

    # num_cv_runs = 10
    # num_cv_spits = 10
    #
    # StratifiedKFold object
    # cv = KFold(CV_SPLITS, True)
    # cv = StratifiedKFold(num_cv_splits, True)
    #
    # for train, test in cv.split(features, labels.argmax(axis=1)):
    #     split_num += 1
    #     logging.info('CV Split {0:d}'.format(split_num))
    #
    #     if split_num == num_cv_runs:
    #         break

    # Test train split.
    if encode_labels_needed:
        X_train, X_test, y_train, y_test = train_test_split(docs, y_bin,
                                                            test_size=0.1,
                                                            random_state=1234,
                                                            stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(docs, y_bin,
                                                            test_size=0.1,
                                                            random_state=1234)

    # Create and train nn.
    print("building text_cnn")

    nn = cnn.GaoTextCNN(vocab, num_classes, max_words)

    nn.train(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))


    # ================================ +
    #                                 /
    #    T R A I N  N E T W O R K    /
    #                               /
    # ---------------------------- +


    # =============== +
    #                /
    #    METRICS    /
    #              /
    # ----------- +

if __name__ == '__main__':
    main(sys.argv[1:])
