#! /usr/bin/env python

# Wikimedia Toxicity Personal Attacks Data Prep.

import sys
import os
from pathlib import Path
import getopt
import logging
import numpy as np
import pandas as pd
import msgpack
import re
import csv
from gensim.models import Word2Vec
from io import BytesIO
from tensorflow.python.lib.io import file_io


def main(argv):

    # ======================= +
    #                        /
    #    D E F A U L T S    /
    #                      /
    # ------------------- +

    # Default log level.
    logging.basicConfig(level=logging.INFO)

    # Default data directory.
    data_dir = ''

    # Default number of votes to decide label based on the annotations of the
    # ten workers who annotated the dataset. Greater than threshold causes a
    # label of true.
    threshold = 5

    # ================================= +
    #                                  /
    #    P A R S E  C L I  A R G S    /
    #                                /
    # ----------------------------- +

    # TODO Take arg for size of embeddings.
    try:
        opts, args = getopt.getopt(argv, "m:d:", ['threshold=', 'data_path='])
    except getopt.GetoptError:
        print('data_prep_WM_PA_Gao_300.py -t <threshold> -d <data_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('data_prep_WM_PA_Gao_300.py\n'
                  + 'Defaults:\n'
                  + '  threshold=\t\tMin number of annotations for label to be true\n'
                  + '  data_path=\t\tData path')
            sys.exit()
        elif opt == '-v':
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == '-t':
            threshold = int(arg)
        elif opt == '-d':
            data_dir = str(arg)

    # Set vars with default or passed-in values.

    # Path to data directory.
    data_path = Path(data_dir)

    # Output path.
    output_path = Path('output')

    # ======================== +
    #                         /
    #    R E A D  D A T A    /
    #                       /
    # -------------------- +

    # Get the data, create dataframes from the tab-separated files.
    attacks_comments_path = data_path / 'attack_annotated_comments.tsv'
    attacks_comments_df = pd.read_csv(attacks_comments_path, sep='\t', header=0)

    attacks_labels_path = data_path / 'attack_annotations.tsv'
    attacks_labels_df = pd.read_csv(attacks_labels_path, sep='\t', header=0)

    logging.debug(attacks_comments_df.head())
    logging.debug(attacks_labels_df.head())

    # ============================== +
    #                               /
    #    P R E P A R E  D A T A    /
    #                             /
    # -------------------------- +

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
    attacks = attacks_merged_summed.loc[attacks_merged_summed['attack'] > threshold].copy()
    attacks.reset_index(level=0, inplace=True)
    attacks['attack'] = 1
    attacks.drop(
        ['year', 'logged_in', 'worker_id', 'quoting_attack', 'recipient_attack',
         'third_party_attack', 'other_attack'], axis=1, inplace=True)

    # Build set of rev_ids that do not contain attacks.
    no_attacks = attacks_merged_summed.loc[
        attacks_merged_summed['attack'] <= threshold].copy()
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
    csv_path = Path('raw_features.csv')
    features.to_csv(csv_path)

    logging.debug(features.head())

    # =================================================== +
    #                                                    /
    #    P R E P A R E  W O R D  E M B E D D I N G S    /
    #                                                  /
    # ----------------------------------------------- +

    # Build vocabulary and word embeddings from source.

    # Store records
    labels = []
    tokens = []
    rev_ids = []
    maxsentlen = 0
    maxdoclen = 0
    num_dropped = 0

    # Process csv one line at a time
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        lineno = 0
        idx = 0
        for line in csv_reader:
            # Skip header.
            lineno += 1
            sys.stdout.write("Processing line %i     \r" % lineno)
            sys.stdout.flush()

            # Begin at index 1.
            idx += 1

            text = line['comment']

            # Process raw text.

            # Remove unwanted tokens.
            text = re.sub('NEWLINE_TOKEN', ' ', text)
            text = re.sub('TAB_TOKEN', ' ', text)

            # Force lowercase.
            text = text.lower()

            # Remove single and double backticks.
            text = re.sub("`", '', text)

            # Remove single quotes.
            text = re.sub("'", '', text)

            # Replace multiple periods in sequence with one period.
            text = re.sub("\.{2,}", '.', text)

            # Replace everything except words, '.', '|', '?', and '!' with space.
            text = re.sub('[^\w_|\.|\?|!]+', ' ', text)

            # Replace periods with ' . '.
            text = re.sub('\.', ' . ', text)

            # Replace '?' with ' ? '.
            text = re.sub('\?', ' ? ', text)

            # Replace '!' with ' ! '.
            text = re.sub('!', ' ! ', text)

            # Tokenize by splitting on whitespace.
            # No leading or trailing whitespace is kept.
            # Consecutive spaces are treated as a single space.
            text = text.split()

            # Drop empty comments.
            if len(text) == 0:
                num_dropped += 1
                continue

            # Split into sentences.
            sentences = []
            sentence = []
            for t in text:
                # Use '.', '!', '?' as markers of end of sentence.
                if t not in ['.', '!', '?']:
                    # Not at end of a sentence.
                    sentence.append(t)
                else:
                    # At end of a sentence.
                    sentence.append(t)

                    # Add sentence to sentences.
                    sentences.append(sentence)

                    # Track longest sentence.
                    if len(sentence) > maxsentlen:
                        maxsentlen = len(sentence)

                    # Reset sentence list.
                    sentence = []

            # If sentence has word, add to list of sentences.
            if len(sentence) > 0:
                sentences.append(sentence)

            # Add split sentences to tokens.
            tokens.append(sentences)

            # Track longest document.
            if len(sentences) > maxdoclen:
                maxdoclen = len(sentences)

            # Add label.
            labels.append(line['attack'])

            # Add rev_id.
            rev_ids.append(line['rev_id'])

    # Use all processed raw text to train word2vec.
    allsents = [sent for doc in tokens for sent in doc]
    # TODO Make embedding size a cli arg w/ default of 200 or 300.
    embedding_size = 300
    model = Word2Vec(allsents, min_count=5, size=embedding_size, workers=4,
                     iter=5)
    model.init_sims(replace=True)

    # Save all word embeddings to matrix
    vocab = np.zeros((len(model.wv.vocab) + 1, embedding_size))
    word2id = {}

    # First row of embedding matrix isn't used so that 0 can be masked.
    for key, val in model.wv.vocab.items():
        # Begin indexes with offset of 1.
        idx = val.__dict__['index'] + 1

        # Build 2D np array (idx, vector)
        vocab[idx, :] = model[key]

        # Dictionary mapping word to index.
        word2id[key] = idx

    # Switch keys/values and store id2word dictionary.
    # Needed to decode examples.
    id2word = {y: x for x, y in word2id.items()}

    # Normalize embeddings.
    vocab -= vocab.mean()
    vocab /= (vocab.std() * 2)

    # Reset first row to 0.
    vocab[0, :] = np.zeros(embedding_size)

    # Add additional word embedding for unknown words.
    vocab = np.concatenate((vocab, np.random.rand(1, embedding_size)))

    # Index for unknown words.
    unk = len(vocab) - 1

    # Convert words to word indices.
    data = {}
    for idx, doc in enumerate(tokens):
        sys.stdout.write(
            'processing %i of %i records       \r' % (idx + 1, len(tokens)))
        sys.stdout.flush()
        dic = {}

        # Get label for each index.
        dic['label'] = labels[idx]

        # Get text of each document.
        dic['text'] = doc

        # Build list of indicies representing the words of each sentence,
        # if word is not a key in word2id mapping, use unk:: vocab[len(vocab)-1].
        indicies = []
        for sent in doc:
            indicies.append(
                [word2id[word] if word in word2id else unk for word in sent])

        # Add indices to dictionary.
        dic['idx'] = indicies

        # Add rev_id to dictionary.
        dic['rev_id'] = rev_ids[idx]

        # Add dictionary containing label, text, indices to data dictionary at index.
        data[idx] = dic

    # Create output dir if it doesn't exist.
    try:
        os.makedirs(output_path)
    except FileExistsError:
        logging.info('Output directory already exists.')

    # Write data dictionary to file.
    date_filename = 'WM-PA-Bin-Threshold-' + str(threshold) + '-Gao-300-data.bin'
    data_path = output_path / date_filename
    with open(data_path, 'wb') as f:
        msgpack.pack(data, f)

    # Write embeddings to file in numpy binary format.
    embeddings_path = output_path / 'WM-PA-300-embeddings'
    np.save(embeddings_path, vocab)

    id2word_path = output_path / 'WM-PA-Gao-300-id2word.bin'
    with open(id2word_path, 'wb') as f:
        msgpack.pack(id2word, f)


if __name__ == '__main__':
    main(sys.argv[1:])
