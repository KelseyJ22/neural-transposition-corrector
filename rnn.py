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

from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from updated import UpdatedModel
from defs import LBLS
from gru_cell import GRUCell

logger = logging.getLogger('final.rnn')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    max_sentence_length = 10
    vocab_size = 2002
    n_classes = vocab_size
    dropout = 0.5
    characters = 'abcdefghijklmnopqrstuvwxyz'
    charset_size = len(characters)
    embed_size = 3 * charset_size
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001


    def __init__(self, args):
        self.cell = args.cell

        if 'output_path' in args:
            self.output_path = args.output_path
        else:
            self.output_path = 'results/{}/{:%Y%m%d_%H%M%S}/'.format(self.cell, datetime.now())
        self.model_output = self.output_path + 'model.weights'
        self.eval_output = self.output_path + 'results.txt'
        self.conll_output = self.output_path + '{}_predictions.conll'.format(self.cell)
        self.log_output = self.output_path + 'log'


def pad_sequences(data):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    Args:
        data: is a list of (sentence, labels) tuples.
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    zero_vector = [0]
    zero_label = 4 # corresponds to the 'O' tag

    for sentence, labels in data:
        new_sentence = list()
        new_labels = list()
        mask = list()
        i = 0

        while len(new_sentence) < Config.max_sentence_length:
            if i < len(sentence): # still in the sentence
                new_sentence.append(sentence[i])
                new_labels.append(labels[i])                
                mask.append(True)
            else: # pad with zeros
                new_sentence.append(zero_vector)
                new_labels.append(zero_label)
                mask.append(False)
            i += 1

        ret.append((new_sentence, new_labels, mask))
    return ret


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
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        # TODO: make this work with my version of embeddings!
        e1 = tf.Variable(self.pretrained_embeddings)
        e2 = tf.nn.embedding_lookup(e1, self.input_placeholder)
        embeddings = tf.reshape(e2, [-1, self.config.max_length, self.config.n_features*self.config.embed_size])

        return embeddings


    def add_prediction_op(self):
        """
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        x = self.add_embedding()

        preds = list()
      
        cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)

        U = tf.get_variable('U', shape=[self.config.hidden_size, self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', shape=[self.config.n_classes,], initializer = tf.constant_initializer(0))
        h_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size])

        with tf.variable_scope('RNN'):
            for time_step in range(self.config.max_sentence_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                o_t, h_t = cell(x[:,time_step,:], h_t)
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b2
                preds.append(y_t)

        preds = tf.pack(preds, 1) # TODO: read up on what this does

        assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes], 'predictions are not of the right shape. Expected {}, got {}'.format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds


    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        Args:
            preds: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
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
        # TODO what should this do?
        """ def featurize_windows(data, start, end, window_size = 1):
            # Uses the input sequences in @data to construct new windowed data points.
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.helper.START, self.helper.END)
        return pad_sequences(examples, self.max_length)"""


    def consolidate_predictions(self, examples_raw, examples, preds):
        # Batch the predictions into groups of sentence length.
        # TODO: what are should this do?
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m]
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret


    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed) # TODO: why axis=2?
        return predictions


    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch, dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.pretrained_embeddings = pretrained_embeddings

        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()


def train(args):
    config = Config(args)
    # TODO: what is helper?
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    helper.save(config.output_path)
    train, test, word_to_id, id_to_word = utils.load_from_file()

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None

    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = RNNModel(helper, config)
        logger.info('took %.2f seconds', time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw)) # TODO: how are dev_raw and dev different?
                report.save()
            else:
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                # TODO: what is LBLS?
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)


def evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    train, test, word_to_id, id_to_word = utils.load_from_file()
   
    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = RNNModel(helper, config)

        logger.info('took %.2f seconds', time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, train):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)


def shell(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    train, test, word_to_id, id_to_word = utils.load_from_file()

    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = RNNModel(helper, config)
        logger.info('took %.2f seconds', time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome! You can use this shell to explore the behavior of your model.""")
            while True:
                try:
                    sentence = raw_input('input> ')
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ['O'] * len(tokens))]):
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
                except EOFError:
                    print('Closing session.')
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests RNN model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default='Data/train', help='Training data')
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default='Data/test', help='Testing data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='Data/vocab.txt', help='Path to vocabulary file')
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default='Data/wordVectors.txt', help='Path to word vectors file')
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default='Data/test', help='Training data')
    command_parser.add_argument('-m', '--model-path', help='Training data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='Data/vocab.txt', help='Path to vocabulary file')
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default='Data/wordVectors.txt', help='Path to word vectors file')
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help='Training data')
    command_parser.set_defaults(func=evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help='Training data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='Data/vocab.txt', help='Path to vocabulary file')
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default='Data/wordVectors.txt', help='Path to word vectors file')
    command_parser.set_defaults(func=shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)