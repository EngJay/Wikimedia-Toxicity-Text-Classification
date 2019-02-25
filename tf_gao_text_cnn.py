import numpy as np
import tensorflow as tf
import sys
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import logging


class GaoTextCNN(object):
    """
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num classes: int
        number of output classes
      - max_words: int
        maximum number of words per document
      - num_filters: int (default: 300)
        number of CNN filters to use
      - dropout_keep: float (default: 0.5)
        dropout keep rate for final softmax layer

    methods:
      - train(data, labels, epochs=30, savebest=False, filepath=None)
        train network on given data
      - predict(data)
        return the one-hot-encoded predicted labels for given data
      - score(data,labels,bootstrap=False,bs_samples=100)
        return the accuracy of predicted labels on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
    """

    def __init__(self, embedding_matrix, num_classes,
                 max_words, num_filters=300, dropout_keep=0.5):

        self.vocab = embedding_matrix

        # Number of columns in embedding matrix.
        self.embedding_size = embedding_matrix.shape[1]

        # Copy the array of embeddings as a matrix of np.float32 elements.
        self.embeddings = embedding_matrix.astype(np.float32)

        # Maximum number of word per document.
        self.mw = max_words

        # Dropout keep rate for final softmax layer.
        self.dropout_keep = dropout_keep

        # TensorFlow placeholder to which to feed dropout rate.
        self.dropout = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Doc input and mask.
        # TensorFlow placeholder to which to feed each document
        # in array of length max_words.
        self.doc_input = tf.placeholder(tf.int32, shape=[max_words], name="input_x")

        # Fancy way of finding number of words in doc.
        #   tf.sign() returns 1 element-wise if element is greater than 0.
        #   Then, tf.reduce_sum() collapses the 1D array into a single
        #   number by addition.
        self.num_words = tf.reduce_sum(tf.sign(self.doc_input))

        #
        self.doc_input_reduced = tf.expand_dims(self.doc_input[:self.num_words], 0)

        # Word embeddings.
        self.word_embeds = tf.gather(
            tf.get_variable('embeddings', initializer=self.embeddings,
                            dtype=tf.float32), self.doc_input_reduced)

        # Convolutional layers.



        with tf.name_scope("conv-maxpool-3"):
            conv3 = tf.layers.conv1d(self.word_embeds, num_filters, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.orthogonal_initializer())
            pool3 = tf.reduce_max(conv3, 1)

        with tf.name_scope("conv-maxpool-4"):
            conv4 = tf.layers.conv1d(self.word_embeds, num_filters, 4, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.orthogonal_initializer())
            pool4 = tf.reduce_max(conv4, 1)

        with tf.name_scope("conv-maxpool-5"):
            conv5 = tf.layers.conv1d(self.word_embeds, num_filters, 5, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.orthogonal_initializer())
            pool5 = tf.reduce_max(conv5, 1)

        # Concatenate.
        self.doc_embed = tf.concat([pool3, pool4, pool5], 1)

        with tf.name_scope("dropout"):
            self.doc_embed = tf.nn.dropout(self.doc_embed, self.dropout)

        # Classification functions.
        # TODO Do something with the huge chunk of stuff.
        self.output = tf.matmul(self.doc_embed,
                                tf.get_variable('W_softmax',
                                                (num_filters * 3, num_classes),
                                                tf.float32,
                                                tf.orthogonal_initializer())) + \
                      tf.get_variable('b_softmax', (num_classes), tf.float32,
                                      tf.zeros_initializer())

        self.prediction = tf.nn.softmax(self.output, name="prediction")
        tf.summary.histogram("prediction", self.prediction)

        # Loss, accuracy, and training functions.
        self.labels = tf.placeholder(tf.float32, shape=[num_classes], name="input_y")
        self.labels_rs = tf.expand_dims(self.labels, 0)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.output, labels=self.labels_rs))
        tf.summary.histogram("loss", self.loss)
        self.optimizer = \
            tf.train.AdamOptimizer(0.00001, 0.9, 0.99).minimize(self.loss)

        # Init op.
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # Initialize global variables by
        self.sess.run(self.init_op)

    def _list_to_numpy(self, inputval):
        """
        convert variable length lists of input values to zero padded numpy array
        """
        if type(inputval) == list:
            retval = np.zeros(self.mw)
            for i, word in enumerate(inputval):
                retval[i] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")

    def train(self, data, labels, epochs=30,
              validation_data=None, savebest=False, filepath=None):
        """
        train network on given data

        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - epochs: int (default: 30)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - savebest: boolean (default: False)
            set to True to save the best model based on validation score per epoch
          - filepath: string (optional)
            path to save model if savebest is set to True

        outputs:
            None
        """
        if savebest == True and filepath == None:
            raise Exception("Please enter a path to save the network")

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validating on %i documents' \
              % (len(data), validation_size))

        with self.sess as sess:
            # Output directory for models and summaries
            timestamp = str(int(time.time()))

            # Track best model for saving.
            prevbest = 0
            for i in range(epochs):
                # TODO FEATURE Add gathering of stats for confusion matrix.
                correct = 0
                y_pred = []
                y_true = []
                start = time.time()

                # Train.
                counter = 0
                for doc in range(len(data)):
                    counter += 1
                    #merge = tf.summary.merge_all()

                    inputval = self._list_to_numpy(data[doc])
                    feed_dict = {self.doc_input: inputval, self.labels: labels[doc],
                                 self.dropout: self.dropout_keep}
                    pred, cost, _ = self.sess.run(
                        [self.prediction, self.loss, self.optimizer],
                        feed_dict=feed_dict)

                    # Collect raw stats for calculating metrics.
                    if np.argmax(pred) == np.argmax(labels[doc]):
                        correct += 1

                    # Collect predictions for calculating metrics with sklearn.
                    # Build array of y_pred.
                    # Insert each prediction at the same index of its label
                    # in the y_true array.
                    y_pred.insert(doc, np.argmax(pred))
                    y_true.insert(doc, np.argmax(labels[doc]))

                    sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r" \
                                     % (i + 1, doc + 1, len(data), cost))
                    sys.stdout.flush()

                    if (doc + 1) % 50000 == 0:
                        score = self.score(validation_data[0], validation_data[1])
                        print("iteration %i validation accuracy: %.4f%%" % (
                        doc + 1, score * 100))

                print()
                # print("training time: %.2f" % (time.time()-start))
                trainscore = correct / len(data)
                print("epoch %i (Gao's) training accuracy: %.4f%%" % (i + 1, trainscore * 100))

                # Log metrics per epoch.
                # TODO Print a clean, well-organized report.
                # TODO Also generate a CSV for easy analysis.
                logging.debug(print('correct:', correct))
                logging.debug(print('total:', counter))

                print(confusion_matrix(y_true, y_pred))
                print(classification_report(y_true, y_pred))
                print('accuracy:', accuracy_score(y_true, y_pred))
                print('precision:', precision_score(y_true, y_pred))
                print('recall:', recall_score(y_true, y_pred))
                print('f1:', f1_score(y_true, y_pred))
                print('log loss:', log_loss(y_true, y_pred))

                # Log ROC Curve.
                fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_true, y_pred)
                fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_true, y_pred)

                # Log AUC Score.
                auc_RF = roc_auc_score(y_true, y_pred)
                auc_LR = roc_auc_score(y_true, y_pred)

                # TODO Produce plot?
                plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
                plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
                plt.plot([0, 1], [0, 1], 'k-', label='random')
                plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
                plt.legend()
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.savefig(str(Path(r'./logs/plots') / timestamp + '-epoch-' + i))
                plt.clf()

                # Validate.
                # TODO Convert this to use CV splitting like
                # the course reviews project.
                if validation_data:
                    score = self.score(validation_data[0], validation_data[1])
                    print("epoch %i validation accuracy: %.4f%%" % (i + 1, score * 100))

                # Save if performance better than previous best.
                if savebest and score >= prevbest:
                    prevbest = score
                    self.save(filepath)



    def predict(self, data):
        """
        return the one-hot-encoded predicted labels for given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data

        outputs:
            numpy array of one-hot-encoded predicted labels for input data
        """
        labels = []
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input: inputval, self.dropout: 1.0}
            prob = self.sess.run(self.prediction, feed_dict=feed_dict)
            prob = np.squeeze(prob, 0)
            one_hot = np.zeros_like(prob)
            one_hot[np.argmax(prob)] = 1
            labels.append(one_hot)

        labels = np.array(labels)
        return labels

    def score(self, data, labels):
        """
        return the accuracy of predicted labels on given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels

        outputs:
            float representing accuracy of predicted labels on given data
        """
        correct = 0.
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input: inputval, self.dropout: 1.0}
            prob = self.sess.run(self.prediction, feed_dict=feed_dict)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct += 1

        # TODO IMPROVE Add calculation of other metrics.
        accuracy = correct / len(labels)
        return accuracy

    def save(self, filename):
        """
        save the model weights to a file

        parameters:
          - filepath: string
            path to save model weights

        outputs:
            None
        """
        self.saver.save(self.sess, filename)

    def load(self, filename):
        """
        load model weights from a file

        parameters:
          - filepath: string
            path from which to load model weights

        outputs:
            None
        """
        self.saver.restore(self.sess, filename)
