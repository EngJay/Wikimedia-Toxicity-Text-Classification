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
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout
from keras.layers import Reshape, concatenate
from keras.layers.merge import concatenate
from sklearn import metrics

# Build paths.
data_path = Path('data')
word_embeddings_path = data_path / 'wikimedia-PA-Gao-200-embeddings.npy'
lex_embeddings_path = data_path / 'PA-Gao-200-lex-V3-204.npy'
decoding_dic_path = data_path / 'PA-Gao-200-id2word.bin'
features_path = Path('data') / 'wikimedia-PA-min-6-votes-data.bin'

max_doc_length = 400
dev_set_size = 0.1

# Read in embeddings.
logging.info('Loading data.')
f = BytesIO(file_io.read_file_to_string(str(word_embeddings_path), binary_mode=True))
vocab = np.load(f)

# Print size of vocab to confirm size.
print('len(vocab):', len(vocab))

# Read in lexicon embeddings.
f = BytesIO(file_io.read_file_to_string(str(lex_embeddings_path), binary_mode=True))
lex_emb = np.load(f)

# Print size of lexicons to confirm size.
print('len(lex_emb):', len(lex_emb))

# Read in id2word for decoding the encoded examples.
f = BytesIO(file_io.read_file_to_string(str(decoding_dic_path), binary_mode=True))
id2word = msgpack.unpack(f, raw=False)

# Visual check to make sure is2word dict is read in properly.
for i in range(0, 10):
    print(i, id2word.get(i))

# Read in features and labels.
f = BytesIO(file_io.read_file_to_string(str(features_path), binary_mode=True))
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
    # TODO The purpose of this has changed - update comments/docs.
    # Convert data to numpy arrays.
    logging.info('Converting data to arrays.')

    # For keeping number of words in longest document in data.
    max_words = 0

    # Create arrays to store docs and labels.
    docs = []
    labels = []

    # Iterate over data to build arrays of docs and labels.
    num_docs = len(raw_data)
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i + 1, num_docs))
        sys.stdout.flush()

        # Get index of document.
        doc = raw_data[i]['idx']

        # Retrieve document from saved data and cast to array.
        doc = [item for sublist in doc for item in sublist]

        # Add document to docs array.
        docs.append(doc)

        # Add label to label array at same index.
        labels.append(raw_data[i]['label'])

        # Track maximum number of words in document.
        if len(doc) > max_words:
            max_words = len(doc)

    del raw_data
    print()

    # TODO Still needed?
    # Label encoder.
    #   Encode labels with value between 0 and n_classes-1,
    #   so for example 1 to 5 star ratings become 0 to 4.
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # TODO Still needed?
    # Binarize labels in one-vs-all fashion if three or more classes.
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(docs, y_bin,
                                                        test_size=dev_set_size)

    # Decode and print first train example.
    print('Decoded first training doc:', indexes_to_text(x_train[0]))

    # Char used as padding elements.
    pad_id = 0

    # Pads all docs to the length of the longest doc using the pad_id char.
    x_train_padded = sequence.pad_sequences(x_train,
                                            maxlen=max_doc_length,
                                            truncating='post',
                                            padding='post',
                                            value=pad_id)

    x_test_padded = sequence.pad_sequences(x_test,
                                           maxlen=max_doc_length,
                                           truncating='post',
                                           padding='post',
                                           value=pad_id)

    print('Unpadded length of first training doc:\t', len(x_train[0]))
    print('Unpadded length of second training doc:\t', len(x_train[1]))
    print('Padded len of first doc:\t', len(x_train_padded[0]))
    print('Padded len of second doc:\t', len(x_train_padded[1]))
    print('x_train shape:\t\t\t', x_train_padded.shape)
    print('x_test shape:\t\t\t', x_test_padded.shape)
    print()
    print(len(x_train) + len(x_test), 'documents each of length',
          max_doc_length, '.')

    # Store pre-truncated/padded lengths of docs.
    x_len_train = np.array([min(len(x), max_words) for x in x_train])
    x_len_test = np.array([min(len(x), max_words) for x in x_test])
    print('Length of original, unpadded train docs:', x_len_train)
    print('Length of original, unpadded test docs:', x_len_test)

    # Oversample training set to compensate for unbalanced labels.
    # sampler = RandomOverSampler()
    # # Flatten labels array to have shape (n_samples, ) on input.
    # x_train, y_train = sampler.fit_resample(x_train_padded, y_train.flatten())

    # TODO Add data loader to create train/test splits.
    # Turn this into a dataset class.

    return x_train_padded, x_len_train, y_train, x_test_padded, x_len_test, y_test


# Get prepared data.
x_train, x_len_train, y_train, x_test, x_len_test, y_test = prepare_data(data)

# Create first embeddings input layer.
vocab_size = len(vocab)
word_embedding_matrix = vocab.astype(np.float32)
word_embedding_size = word_embedding_matrix.shape[1]

# embed1 = Sequential()
input1 = Input(shape=(max_doc_length,), dtype='int32')
embed1 = Embedding(input_dim=vocab_size,
                   output_dim=word_embedding_size,
                   weights=[vocab],
                   input_length=max_doc_length,
                   # mask_zero=True,
                   trainable=False)(input1)

print('embed1.shape:', embed1.shape)

# Create second embeddings input layer.
lex_size = len(lex_emb)
lex_embedding_matrix = lex_emb.astype(np.float32)
lex_embedding_size = lex_embedding_matrix.shape[1]

# Pad lex embeddings to match width of vocab.
embeddings_size_delta = word_embedding_size - lex_embedding_size

print('embeddings_size_delta:', embeddings_size_delta)

# No padding above, below, or to the left,
# padding to the right of the difference between the word and lex matrices.
padded_lex_embedding_matrix = np.pad(lex_embedding_matrix,
                                     ((0, 0), (0, embeddings_size_delta)),
                                     'constant')
padded_lex_embedding_size = padded_lex_embedding_matrix.shape[1]

print('padded_lex_embedding_size.shape', padded_lex_embedding_matrix.shape)

# embed2 = Sequential()
input2 = Input(shape=(max_doc_length,), dtype='int32')
embed2 = Embedding(input_dim=lex_size,
                   output_dim=padded_lex_embedding_size,
                   weights=[padded_lex_embedding_matrix],
                   input_length=max_doc_length,
                   # mask_zero=True,
                   trainable=False)(input2)

print('embed2.shape', embed2.shape)

# Create model that takes two embedding input layers
# and merges them into a single multichannel layer.
merged = concatenate([embed1, embed2])

print('merged.shape', merged.shape)

reshaped = Reshape((max_doc_length, word_embedding_size, 2))(merged)

print('reshaped.shape:', reshaped)

# Convolution 3
conv2D_3 = Conv2D(filters=64, kernel_size=(3, word_embedding_size),
                  activation='relu',
                  border_mode='valid')(reshaped)
print('conv2D_3.shape:', conv2D_3.shape)
max_pool_3 = MaxPooling2D((max_doc_length-3+1, 1), padding='valid')(conv2D_3)
print('max_pool_3:', max_pool_3.shape)

# Convolution 4
conv2D_4 = Conv2D(filters=64, kernel_size=(4, word_embedding_size),
                  activation='relu',
                  border_mode='valid')(reshaped)
print('conv2D_4.shape:', conv2D_4.shape)
max_pool_4 = MaxPooling2D((max_doc_length-4+1, 1), padding='valid')(conv2D_4)
print('max_pool_4:', max_pool_4.shape)

# Convolution 5
conv2D_5 = Conv2D(filters=64, kernel_size=(5, word_embedding_size),
                  activation='relu',
                  border_mode='valid')(reshaped)
print('conv2D_5.shape:', conv2D_5.shape)
max_pool_5 = MaxPooling2D((max_doc_length-5+1, 1), padding='valid')(conv2D_5)
print('max_pool_5:', max_pool_4.shape)

# Concatenate the pools.
h_pool = concatenate([max_pool_3, max_pool_4, max_pool_5])

# Flatten pool.
flattened = Flatten()(h_pool)

print('flattened:', flattened.shape)

dropout1 = Dropout(rate=1-0.5)(flattened)

dense = Dense(256, activation="relu")(dropout1)

print('dense.shape:', dense.shape)

dropout2 = Dropout(rate=1-0.5)(dense)

softmax_pred = Dense(2, activation="softmax")(dropout2)

model = Model(inputs=[input1, input2], outputs=softmax_pred)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

# Training data must be provided to both inputs.
model.fit(x=[x_train, x_train], y=y_train, batch_size=128, epochs=1,
          validation_data=([x_test, x_test], y_test))

y_pred_prob = model.predict([x_test, x_test])

y_pred = np.argmax(y_pred_prob, axis=1)

pred_results = {'f1_macro': metrics.f1_score(y_test, y_pred, average='macro')}

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

pred_results['tn'] = tn
pred_results['fp'] = fp
pred_results['fn'] = fn
pred_results['tp'] = tp
pred_results['auc'] = metrics.roc_auc_score(y_test, y_pred)
pred_results['acc'] = metrics.accuracy_score(y_test, y_pred)
pred_results['precision'] = metrics.precision_score(y_test, y_pred)
pred_results['recall'] = metrics.recall_score(y_test, y_pred)
pred_results['f1'] = metrics.f1_score(y_test, y_pred)
pred_results['log_loss'] = metrics.log_loss(y_test, y_pred_prob)

print('Prediction Results:', pred_results)

exp_num = 1
exp_name = 'Shin-multichannel-Gao-200-w-lex'
csv_path = Path('results')

# TODO Pull this CSV writing stuff out into a function.
# Build filename of CSV.
csv_name = Path('results-' + exp_name + '-exp-' + str(exp_num) + '.csv')

# Build path to which to write CSV.
csv_path = Path(csv_path / csv_name)

# Check for existence of csv to determine if header is needed.
results_file_exists = os.path.isfile(str(csv_path))

# Open file, write/append results.
with open(str(csv_path), mode='a') as csv_file:
    # fieldnames = list(eval_results.keys())
    fieldnames = ['log_loss', 'acc', 'precision',
                  'recall', 'auc', 'f1', 'f1_macro',
                  'tp', 'tn', 'fp', 'fn']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header only if results csv did not exist at beginning
    # of this trip through.
    if not results_file_exists:
        writer.writeheader()

    # Write row for each run.
    writer.writerow(pred_results)
