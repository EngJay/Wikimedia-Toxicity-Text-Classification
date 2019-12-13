import sys
import re
from pathlib import Path
import csv
import logging
import numpy as np
from gensim.models import Word2Vec
import msgpack


def extract_features(csv_path):
    # df = pd.read_csv(csv_path)
    # logging.info(print('Number of records:', len(df)))
    # logging.debug(print(df.head()))

    # Tokens to remove.
    # TODO Make this a setting.
    remove_tokens = ['NEWLINE_TOKEN', ]

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
            text = line['comment']

            # Process raw text.

            # Force lowercase.
            text = text.lower()

            # Remove unwanted tokens.
            text = re.sub("newline_token", '', text)
            text = re.sub("tab_token", '', text)

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
            labels.append(line['attack'])

    # Use all processed raw text to train word2vec.
    # TODO Incorporate FastText or other algos, too.
    allsents = [sent for doc in tokens for sent in doc]
    # TODO Make embedding size a cli arg w/ default of 200 or 300.
    embedding_size = 200
    model = Word2Vec(allsents, min_count=5, size=embedding_size, workers=4, iter=5)
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
    data_path = Path(r'data') / 'cache' / 'wikimedia-personal-attacks' / \
        'wikimedia-personal-attacks-data.bin'
    with open(data_path, 'wb') as f:
        msgpack.pack(data, f)

    # Write embeddings to file in numpy binary format.
    # TODO Take arg for filename and path.
    embeddings_path = Path(r'data') / 'cache' / 'wikimedia-personal-attacks' / \
        'wikimedia-personal-attacks-embeddings'
    np.save(embeddings_path, vocab)
