
# TODO Data prep script for wikimedia personal attack dataset.

#! /usr/bin/env python

# Wikimedia Toxicity Personal Attacks Data Prep Script.

# TODO Finish adjusting this for the attack dataset.

import sys
from pathlib import Path
import getopt
import logging
import numpy as np
import pandas as pd
import msgpack
import re
import csv
from gensim.models import Word2Vec


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
    data_dir = r''

    # Default number of record to prepare -1 = all records.
    num_records = -1

    # Default number of votes to decide label based on the annotations of the
    # ten workers who annotated the dataset.
    min_num_votes = 6

    # ================================= +
    #                                  /
    #    P A R S E  C L I  A R G S    /
    #                                /
    # ----------------------------- +

    # Parse cli args.
    # TODO Clean this up.
    try:
        opts, args = getopt.getopt(argv, "m:n:", ["min_num_votes=", "num_records="])
    except getopt.GetoptError:
        print('train.py -m <min_num_votes> -n <num_records>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -n <num_records>\n'
                  + 'Defaults:\n'
                  + '  min_num_votes=\t\tMin number of votes\n'
                  + '  num_records=\t\tNumber of records\n')
            sys.exit()
        elif opt == '-v':
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == '-m':
            min_num_votes = int(arg)
        elif opt == '-n':
            num_records = int(arg)

    # Set vars with default or passed-in values.

    # Path to data directory.
    data_path = Path(data_dir)

    # ============================== +
    #                               /
    #    P R E P A R E  D A T A    /
    #                             /
    # --------------------------- +

    # Get the data, create dataframes from the tab-separated files.
    attacks_comments_path = data_path / 'attack_annotated_comments.tsv'
    attacks_comments_df = pd.read_csv(attacks_comments_path, sep='\t', header=0)

    attacks_labels_path = data_path / 'attack_annotations.tsv'
    attacks_labels_df = pd.read_csv(attacks_labels_path, sep='\t', header=0)

    logging.debug(attacks_comments_df.head())
    logging.debug(attacks_labels_df.head())

    # Build paths to word embeddings.
    # TODO IMPROVEMENT Make this a cli arg w/ Google vectors as default.
    raw_embeddings_filename = data_path / 'pretrained_embeddings' / \
                                            'GoogleNews-vectors-negative300.bin'
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
    raw_features_path = Path('raw_features.csv')
    features.to_csv(raw_features_path)

    logging.debug(print(features.head()))

    # =================================================== +
    #                                                    /
    #    P R E P A R E  W O R D  E M B E D D I N G S    /
    #                                                  /
    # ----------------------------------------------- +

    # Build vocabulary and word embeddings from source if needed.

    extract_features(raw_features_path)


def extract_features(csv_path):
    # Store records
    labels = []
    tokens = []
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
            logging.debug(print("Processing line %i     \r" % lineno))

            # Begin at index 1.
            idx += 1

            # TODO This is coupled to this field. Change to arg?
            text = line['comment_text']

            # Process raw text.

            # Force lowercase.
            text = text.lower()

            # Remove unwanted tokens.
            text = re.sub("\\n", ' ', text)
            text = re.sub("\\t", ' ', text)

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

            # Drop empty reviews.
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

            # Add label
            # Only adding the 'insult' labels for comparison to the personal attacks
            # attacks classification of the raw wikimedia data.
            label = []
            # label.append(line['toxic'])
            # label.append(line['severe_toxic'])
            # label.append(line['obscene'])
            # label.append(line['threat'])
            label.append(line['insult'])
            # label.append(line['identity_hate'])
            labels.append(label)

    # Use all processed raw text to train word2vec.
    # TODO Incorporate FastText or other algos, too.
    allsents = [sent for doc in tokens for sent in doc]
    # TODO Make embedding size a cli arg w/ default of 200 or 300.
    embedding_size = 200
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

    # Normalize embeddings.
    vocab -= vocab.mean()
    vocab /= (vocab.std() * 2)

    # Reset first row to 0.
    vocab[0, :] = np.zeros((embedding_size))

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
        # if word is a key in word2id mapping, use unk, defined: vocab[len(vocab)-1].
        indicies = []
        for sent in doc:
            indicies.append(
                [word2id[word] if word in word2id else unk for word in sent])

        # Add indices to dictionary.
        dic['idx'] = indicies

        # Add dictionary containing label, text, indices to data dictionary at index.
        data[idx] = dic

    # Write data dictionary to file.
    # TODO Take arg for filename and path.
    data_path = Path(r'prepared') / 'jigsaw-wikimedia-data.bin'
    with open(data_path, 'wb') as f:
        msgpack.pack(data, f)

    # Write embeddings to file in numpy binary format.
    # TODO Take arg for filename and path.
    embeddings_path = Path(r'prepared') / 'jigsaw-wikimedia-embeddings'
    np.save(embeddings_path, vocab)


if __name__ == '__main__':
    main(sys.argv[1:])
