#! /usr/bin/env python

# Wikimedia Toxicity Personal Attacks Multilabel Data Prep.

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
        print('data_prep_WM_PA_multilabel.py -t <threshold> -d <data_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('data_prep_WM_PA_multilabel.py\n'
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
    attacks_merged_summed.drop(['year', 'logged_in', 'worker_id'], axis=1, inplace=True)
    
    # Cast floats to ints to avoid later processing errors in TF/Keras.
    attacks_merged_summed.quoting_attack = pd.to_numeric(
        attacks_merged_summed.quoting_attack, 
        downcast='integer')
    attacks_merged_summed.recipient_attack = pd.to_numeric(
        attacks_merged_summed.recipient_attack, 
        downcast='integer')
    attacks_merged_summed.third_party_attack = pd.to_numeric(
        attacks_merged_summed.third_party_attack, 
        downcast='integer')
    attacks_merged_summed.other_attack = pd.to_numeric(
        attacks_merged_summed.other_attack, 
        downcast='integer')
    attacks_merged_summed.attack = pd.to_numeric(
        attacks_merged_summed.attack, 
        downcast='integer')
    
    # Despite using the same threshold as the binary version of 
    # this dataset, 5, reducing the number of less-frequent labels 
    # quite low, it's retained for a better comparison between the models. 
    threshold = 5
    print('With threshold:', threshold)
    print('Num quoting_attack:\t',
          len(attacks_merged_summed[attacks_merged_summed['quoting_attack'] > threshold]))
    print('Num recipient_attack:\t',
          len(attacks_merged_summed[attacks_merged_summed['recipient_attack'] > threshold]))
    print('Num third_party_attack:\t',
          len(attacks_merged_summed[attacks_merged_summed['third_party_attack'] > threshold]))
    print('Num other_attack:\t',
          len(attacks_merged_summed[attacks_merged_summed['other_attack'] > threshold]))
    print('Num attack:\t\t',
          len(attacks_merged_summed[attacks_merged_summed['attack'] > threshold]))
    print()
    
    # Set labels for each col by keeping values that match condition, 
    # replace values that don't match with other. 
    attacks_merged_summed.quoting_attack.where(
        attacks_merged_summed.quoting_attack > threshold, other=0, inplace=True)
    attacks_merged_summed.quoting_attack.where(
        attacks_merged_summed.quoting_attack <= threshold, other=1, inplace=True)
    
    attacks_merged_summed.recipient_attack.where(
        attacks_merged_summed.recipient_attack > threshold, other=0, inplace=True)
    attacks_merged_summed.recipient_attack.where(
        attacks_merged_summed.recipient_attack <= threshold, other=1, inplace=True)

    attacks_merged_summed.third_party_attack.where(
        attacks_merged_summed.third_party_attack > threshold, other=0, inplace=True)
    attacks_merged_summed.third_party_attack.where(
        attacks_merged_summed.third_party_attack <= threshold, other=1, inplace=True)

    attacks_merged_summed.other_attack.where(
        attacks_merged_summed.other_attack > threshold, other=0, inplace=True)
    attacks_merged_summed.other_attack.where(
        attacks_merged_summed.other_attack <= threshold, other=1, inplace=True)
    
    # Check to make sure the labels match the original estimate.
    num_quoting_attack = len(
        attacks_merged_summed.quoting_attack[attacks_merged_summed.quoting_attack == 1])
    num_recipient_attack = len(
        attacks_merged_summed.recipient_attack[attacks_merged_summed.recipient_attack == 1])
    num_third_party_attack = len(
        attacks_merged_summed.third_party_attack[attacks_merged_summed.third_party_attack == 1])
    num_other_attack = len(
        attacks_merged_summed.other_attack[attacks_merged_summed.other_attack == 1])

    print('Num labels should match the threshold check.')    
    print('Num quoting_attack:\t', num_quoting_attack)
    print('Num recipient_attack:\t', num_recipient_attack)
    print('Num third_party_attack:\t', num_third_party_attack)
    print('Num other_attack:\t', num_other_attack)
    
    # Same treatment of the binary attack label.
    attacks_merged_summed.attack.where(
        attacks_merged_summed.attack > threshold, other=0, inplace=True)
    attacks_merged_summed.attack.where(
        attacks_merged_summed.attack <= threshold, other=1, inplace=True)
          
    # This should match the binary label dataset.
    num_attack = len(attacks_merged_summed.attack[attacks_merged_summed.attack == 1])
    print('Num attack:\t', num_attack)
          
    # Breakdown of balance of dataset.
    print()
    print('Percentage of examples with attacks:')
    total_num = len(attacks_merged_summed)
    print('quoting_attack:\t\t', num_quoting_attack / total_num * 100)
    print('recipient_attack:\t', num_recipient_attack / total_num * 100)
    print('third_party_attack:\t', num_third_party_attack / total_num * 100)
    print('other_attack:\t\t', num_other_attack / total_num * 100)
    print('attack:\t\t\t', num_attack / total_num * 100)
          
    # Merge labels with comments on rev_id.
    multilabel_attacks = pd.merge(attacks_comments_df, attacks_merged_summed, 
                              on='rev_id')
    logging.debug(multilabel_attacks.head())
          
    # Drop everything but rev_id, comment, labels, and split.
    multilabel_attacks.drop(['year', 'logged_in', 'ns', 'sample'], axis=1, inplace=True)
          
    # Write dataset to disk.
    csv_path = Path('WM-PA-Multilabel-Min-6-Votes-Dataset.csv')
    multilabel_attacks.to_csv(csv_path)

    # =================================================== +
    #                                                    /
    #    P R E P A R E  W O R D  E M B E D D I N G S    /
    #                                                  /
    # ----------------------------------------------- +
    
    # Build vocabulary and word embeddings from source.

    # Store records
    all_labels = []              
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

            # Build list of labels for record.
            doc_labels = []
            doc_labels.append(line['quoting_attack'])
            doc_labels.append(line['recipient_attack'])
            doc_labels.append(line['third_party_attack'])
            doc_labels.append(line['other_attack'])
            doc_labels.append(line['attack'])
                  
            # Add list of labels to list of all labels.
            all_labels.append(doc_labels)

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

    # Get word2id dictionary.
    # Read in id2word for decoding the encoded examples.
#     f = BytesIO(file_io.read_file_to_string('../Post-Specialized-Embeddings/post-specialisation/results/Gao_300_PA_AR_Post_Spec_word2id.bin',
#                                             binary_mode=True))
#     word2id = msgpack.unpack(f, raw=False)

    # Index for unknown words.
    unk = len(word2id) - 1

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

        # Add rev_id to dictionary.
        dic['rev_id'] = rev_ids[idx]

        # Add dictionary containing label, text, indices to data dictionary at index.
        data[idx] = dic

    # Write data dictionary to file.
    data_output_path = output_path / 'WM-PA-Multilabel-Threshold-5-Gao-data.bin'
    with open(data_output_path, 'wb') as f:
        msgpack.pack(data, f)

    # Write embeddings to file in numpy binary format.
    embeddings_output_path = output_path / 'WM-PA-Multilabel-EMB-Gao-300'
    np.save(embeddings_output_path, vocab)
    
    # # Write id2word dict to file.
    # id2word_output_path = output_path / 'WM-PA-Multilabel-id2word.bin'
    # with open(id2word_output_path, 'wb') as f:
    #     msgpack.pack(id2word, f)


if __name__ == '__main__':
    main(sys.argv[1:])
