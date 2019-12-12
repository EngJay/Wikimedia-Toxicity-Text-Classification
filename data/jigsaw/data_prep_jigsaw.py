#! /usr/bin/env python

# Jigsaw Toxic Comment Classification Challenge Data Prep Script.

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
    
    # Default outout directory.
    output_dir = 'output'

    # ================================= +
    #                                  /
    #    P A R S E  C L I  A R G S    /
    #                                /
    # ----------------------------- +

    # Parse cli args.
    try:
        opts, args = getopt.getopt(argv, "d:o:", ['data_dir=', 'output_dir='])
    except getopt.GetoptError:
        print('data_prep_jigsaw.py -d <data_dir> -o <output_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('data_prep_jigsaw.py -d <data_dir> -o <output_dir>\n'
                  + 'Defaults:\n'
                  + '  data_dir=\t\t'
                  + '  output_dir=\t\toutput')
            sys.exit()
        elif opt == '-v':
            logging.getLogger().setLevel(logging.DEBUG)
        elif opt == '-d':
            data_dir = str(arg)
        elif opt == '-o':
            output_dir = str(arg)

    # Set vars with default or passed-in values.

    # Path to data directory.
    data_path = Path(data_dir)

    # Output path.
    output_path = Path(output_dir)

    # Set vars with default or passed-in values.

    # Get the data, create dataframes from the CSVs.
    train_path = data_path / 'train.csv'
    train_df = pd.read_csv(train_path, header=0)

    test_path = data_path / 'test.csv'
    test_df = pd.read_csv(test_path, header=0)

    test_labels_path = data_path / 'test_labels.csv'
    test_labels_df = pd.read_csv(test_labels_path, header=0)

    # Print heads if debug.
    logging.debug(train_df.head())
    logging.debug(test_df.head())
    logging.debug(test_labels_df.head())

    # ============================== +
    #                               /
    #    P R E P A R E  D A T A    /
    #                             /
    # --------------------------- +

    # Drop everything except for comment_text and the labels.
    train_df.drop(['id'], axis=1, inplace=True)

    # Merge test with test labels.
    merged_test_df = pd.merge(test_df, test_labels_df, on='id')

    # Get list of records with -1 for labels (they weren't used in kaggle evaluation).
                  
    # Drop them.

    # Create output dir if it doesn't exist.
    try:
        os.makedirs(output_path)
    except FileExistsError:
        logging.info('Output directory already exists.')

    # Write features and labels to disk.
    csv_path = output_path / 'raw_train_set.csv'
    train_df.to_csv(csv_path)

    # =================================================== +
    #                                                    /
    #    P R E P A R E  W O R D  E M B E D D I N G S    /
    #                                                  /
    # ----------------------------------------------- +

    # Build vocabulary and word embeddings from source if needed.

    # Store records
    all_labels = []              
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
            sys.stdout.write("Processing line %i     \r" % lineno)
            sys.stdout.flush()

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

            # Build list of labels for record.
            doc_labels = []
            doc_labels.append(line['toxic'])
            doc_labels.append(line['severe_toxic'])
            doc_labels.append(line['obscene'])
            doc_labels.append(line['threat'])
            doc_labels.append(line['insult'])
            doc_labels.append(line['identity_hate'])
                  
            # Add list of labels to list of all labels.
            all_labels.append(doc_labels)

    # Use all processed raw text to train word2vec.
    allsents = [sent for doc in tokens for sent in doc]
    # TODO Make embedding size a cli arg w/ default of 300.
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

    # Switch keys/values and store id2word dictionary (for decoding examples).
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
        dic['labels'] = all_labels[idx]

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
    data_output_path = output_path / 'jigsaw-WM-Gao-data.bin'
    with open(data_output_path, 'wb') as f:
        msgpack.pack(data, f)

    # Write embeddings to file in numpy binary format.
    embeddings_output_path = output_path / 'jigsaw-WM-EMB-Gao-300'
    np.save(embeddings_output_path, vocab)

    # Write id2word dict to file.
    id2word_output_path = output_path / 'jigsaw-WM-EMB-Gao-id2word.bin'
    with open(id2word_output_path, 'wb') as f:
        msgpack.pack(id2word, f)


if __name__ == '__main__':
    main(sys.argv[1:])
