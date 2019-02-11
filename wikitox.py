import os
import sys
from pathlib import Path
import getopt
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Wikimedia Toxicity Text Classification Project

# GPU check.
# Ensure tensorflow can find CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    # Default number of epochs.
    num_epochs = 50

    # Default number of cross-validation splits.
    num_cv_runs = 10
    num_cv_splits = 10

    # ================================= +
    #                                  /
    #    P A R S E  C L I  A R G S    /
    #                                /
    # ----------------------------- +

    # Parse cli args.
    try:
        opts, args = getopt.getopt(argv, "he:c:", ["num_epochs=", "num_cv_splits="])
    except getopt.GetoptError:
        print('train.py -e <num_epochs> -c <num_cv_splits>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -e <num_epochs> -c <num_cv_splits>\n'
                  + 'Defaults:\n'
                  + '  num_epochs=50\t\tNumber of epochs\n'
                  + '  num_cv_splits=50\tNumber of cross-validation splits')
            sys.exit()
        elif opt in ("-e", "--num_epochs"):
            num_epochs = int(arg)
        elif opt in ("-c", "--num_cv_splits"):
            num_cv_runs = int(arg)
            num_cv_splits = int(arg)

    # Set vars with default or passed-in values.
    data_path = Path(r'data')

    # Get the data, create dataframes from the tab-separated files.
    attacks_comments_path = data_path / 'Wikimedia-Toxicity-Personal-Attacks'
    attacks_comments_path = attacks_comments_path / 'attack_annotated_comments.tsv'
    attacks_comments_df = pd.read_csv(attacks_comments_path, sep='\t', header=0)

    attacks_labels_path = data_path / 'Wikimedia-Toxicity-Personal-Attacks'
    attacks_labels_path = attacks_labels_path / 'attack_annotations.tsv'
    attacks_labels_df = pd.read_csv(attacks_labels_path, sep='\t', header=0)

    #print(attacks_comments_df.head())
    #print(attacks_labels_df.head())

    # ============================== +
    #                               /
    #    P R E P A R E  D A T A    /
    #                             /
    # --------------------------- +

    # Concatenate data frames on rev_id.

    # Treat the 10 records for each comment (one for each worker)
    # like votes: > 5 workers reporting a comment contains an attack = 1.


    # Create set of labels.

    # Drop everything but text of comments.


    # Build binary classification CNN.


    # Train the network.


    # Metrics.


if __name__ == '__main__':
    main(sys.argv[1:])
