#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from rnn_model import UpdatedModel
from gru_cell import GRUCell
import utils

logger = logging.getLogger('final.rnn')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    max_sentence_length = 10
    vocab_size = 2393
    n_classes = vocab_size
    dropout = 0.5
    characters = 'abcdefghijklmnopqrstuvwxyz'
    charset_size = len(characters)
    embedding_size = 3 * charset_size
    hidden_size = 800
    batch_size = 32
    n_epochs = 40
    max_grad_norm = 10.
    lr = 0.001
    id_to_word = dict()
    word_to_id = dict()

    def __init__(self, args):
        if 'output_path' in args:
            self.output_path = args.output_path
        else:
            self.output_path = 'results/{:%Y%m%d_%H%M%S}/'.format(datetime.now())
        self.model_output = self.output_path + 'model.weights'
        self.eval_output = self.output_path + 'results.txt'
        self.log_output = self.output_path + 'log'
        if 'model_path' in args:
            self.model_path = args.model_path


class RNNModel(UpdatedModel):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_length))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_length))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.config.max_sentence_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)


    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):        
        feed_dict = {self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch
        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch

        return feed_dict


    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: numpy array of shape (None, config.max_sentence_length, config.embedding_size)
        """
        L = tf.Variable(self.pretrained_embeddings)
        lookups = tf.nn.embedding_lookup(L, self.input_placeholder)
        embeddings = tf.reshape(lookups, [-1, self.config.max_sentence_length, self.config.embedding_size])

        return embeddings


    def add_prediction_op(self):
        """
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        x = tf.cast(self.add_embedding(), tf.float32)
      
        cell = GRUCell(self.config.embedding_size, self.config.hidden_size)

        U = tf.get_variable('U', shape=[self.config.hidden_size, self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=[self.config.n_classes,], initializer = tf.constant_initializer(0))
        h_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size])

        preds = list()
        with tf.variable_scope('RNN'):
            for time_step in range(self.config.max_sentence_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                o_t, h_t = cell(x[:,time_step,:], h_t)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b2
                preds.append(y_t)

        preds = tf.stack(preds, 1) # converts from list to tensor

        assert preds.get_shape().as_list() == [None, self.config.max_sentence_length, self.config.n_classes], 'predictions are not of the right shape. Expected {}, got {}'.format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds


    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        Args:
            preds: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=preds)
        masked = tf.boolean_mask(loss_vector, self.mask_placeholder)
        loss = tf.reduce_mean(masked)
        return loss


    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op


    def preprocess_sequence_data(self, examples):
        x = list()
        y = list()
        for sentence, labels in examples:
            sent_list = list()
            label_list = list()
            assert len(sentence) == len(labels)

            for i in range(0, len(sentence)):
                #word = sentence[i]
                label = labels[i]
                # with current configuration of embedding using label for both works
                # (will need to change if/when I try other embeddings)
                sent_list.append(self.config.word_to_id[label])
                label_list.append(self.config.word_to_id[label])

            assert len(sent_list) == len(label_list)
            if len(sent_list) > 0: # don't want any data to be [], []
                x.append(np.asarray(sent_list))
                y.append(np.asarray(label_list))

        return (utils.pad_sequences(np.asarray(x), np.asarray(y)))


    def consolidate_predictions(self, inputs, masks, preds):
        assert len(inputs) == len(preds) == len(masks)
        correct = 0
        total = 0
        complete = 0
        trivial = 0
        for i in range(0, len(preds)):
            for j in range(0, len(preds[0])):
                if masks[i][j]:
                    if inputs[i][j] == preds[i][j]:
                        correct += 1
                    total += 1
                else:
                    trivial += 1
                complete += 1

        print 'correct', correct, 'out of', total, 'or', correct + trivial, 'out of', complete

        return inputs, masks, preds


    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions


    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def __init__(self, config, pretrained_embeddings):
        super(RNNModel, self).__init__(config)

        self.pretrained_embeddings = pretrained_embeddings

        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()


def lookup_words(predictions, originals, id_to_word):
    sentence_preds = list()
    for pred in predictions:
        sentence = list()
        for word in pred:
            sentence.append(id_to_word[word])
        sentence_preds.append(sentence)

    sentence_in = list()
    for sent in originals:
        sentence = list()
        for word in sent:
            sentence.append(id_to_word[word])
        sentence_in.append(sentence)

    return sentence_preds, sentence_in


def train(args):
    config = Config(args)
    train, test, word_to_id, id_to_word, embeddings = utils.load_from_file_basic()

    config.word_to_id = word_to_id
    config.id_to_word = id_to_word
    utils.save(config.output_path, word_to_id, id_to_word)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = RNNModel(config, embeddings)
        logger.info('took %.2f seconds', time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, test)
            
            sentences, masks, predictions = model.output(session, train)
            originals, predictions = lookup_words(predictions, sentences, id_to_word)
            output = zip(originals, masks, predictions)

            with open('results.txt', 'w') as f:
                utils.save_results(f, output)


def evaluate(args):
    config = Config(args)

    train, test, word_to_id, id_to_word, embeddings = utils.load_from_file_basic()
    config.word_to_id = word_to_id
    config.id_to_word = id_to_word

    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = RNNModel(config, embeddings)

        logger.info('took %.2f seconds', time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_path)

            sentences, masks, predictions = model.output(session, train)
            originals, predictions = lookup_words(predictions, sentences, id_to_word)
            output = zip(originals, masks, predictions)

            with open('eval_results.txt', 'w') as f:
                utils.save_results(f, output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests RNN model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default='../Data/train_all', help='Training data')
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default='../Data/test_all', help='Testing data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='../Data/vocab.txt', help='Path to vocabulary file')
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default='../Data/test_all', help='Training data')
    command_parser.add_argument('-m', '--model-path', help='Training data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='../Data/vocab.txt', help='Path to vocabulary file')
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help='Training data')
    command_parser.set_defaults(func=evaluate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)