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

from window_model import WindowModel
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
    hidden_size = 600
    batch_size = 32
    n_epochs = 40
    max_grad_norm = 10.
    lr = 0.001
    max_word_len = 8
    id_to_word = dict()
    word_to_id = dict()
    word_vec_size = charset_size * max_word_len
    character_embeddings = list()

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


class WindowNN(WindowModel):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_length))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_length))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.config.max_sentence_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)


    def make_windowed_data(self, data):
        # word_vector_len = 26 * max_word_len
        # each chunk of length 26 = 1 in the actual character, 0.5 in the neighboring characters, 0.25 in the +- 2 characters
        # sentence = max_words x word_vector_len

        sentences = list()
        embeddings = list()
        count = 0
        seen = dict()
        for sentence in data:
            sent = list()
            for word in sentence:
                if word not in seen: # if we've encountered this word before we already have an embedding for it
                    word_vec = list()
                    for i in range(0, self.config.max_word_len):

                        if i < len(word):
                            ind = charset.find(word[i])
                            new_vec = [0] * self.config.charset_size
                            new_vec[ind] = 1
                            if i-1 >= 0:
                                ind = self.config.characters.find(word[i-1])
                                new_vec[ind] = 0.5
                            elif i+1 < len(word):
                                ind = self.config.characters.find(word[i+1])
                                new_vec[ind] = 0.5
                            elif i-2 >= 0:
                                ind = self.config.characters.find(word[i-2])
                                new_vec[ind] = 0.25
                            elif i+2 < len(word):
                                ind = self.config.characters.find(word[i+2])
                                new_vec[ind] = 0.25

                            word_vec += new_vec
                        else:
                            word_vec += [0] * self.config.charset_size # padding so all word vectors are the same size

                    seen[word] = count # store this word
                    sent.append(count)
                    embeddings.append(np.asarray(word_vec)) # store this embedding
                    count += 1

                else: # add the already-created id for this word to the sentence
                    num = seen[word]
                    sent.append(num)
            sentences.append(sent)
                  
        return sentences, embeddings


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
        embeddings = tf.Variable(self.config.character_embeddings)
        lookups = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        embedded = tf.reshape(lookups, [-1, self.config.word_vec_size])

        return embedded


    def add_prediction_op(self):
        """
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        x = self.add_embedding()

        W = tf.get_variable('W', shape = [self.config.word_vec_size, self.config.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', initializer = tf.zeros([self.config.hidden_size,]))
        U = tf.get_variable('U', shape = [self.config.hidden_size, self.config.vocab_size], initializer = tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2', initializer = tf.zeros([self.config.vocab_size,]))

        h = tf.nn.relu(tf.matmul(x, W) + b)
        h_drop = tf.nn.dropout(h, self.dropout_placeholder)
        pred = tf.matmul(h_drop, U) + b2
        return pred


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
        sentences, embeddings = make_windowed_data(examples)
        self.config.character_embeddings = embeddings
        return sentences


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
        print labels_batch[0].shape
        feed = self.create_feed_dict(inputs_batch=inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch, dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def __init__(self, config):
        super(WindowNN, self).__init__(config)

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
    train, test, word_to_id, id_to_word, embeddings = utils.load_from_file()

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
        model = WindowNN(config)
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

    train, test, word_to_id, id_to_word, embeddings = utils.load_from_file()
    config.word_to_id = word_to_id
    config.id_to_word = id_to_word

    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = WindowNN(config, embeddings)

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


def shell(args):
    config = Config(args.model_path)
    train, test, word_to_id, id_to_word, embeddings = utils.load_from_file()
    config.word_to_id = word_to_id
    config.id_to_word = id_to_word

    with tf.Graph().as_default():
        logger.info('Building model...',)
        start = time.time()
        model = WindowNN(config, embeddings)
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
                        print sentence, predictions
                except EOFError:
                    print('Closing session.')
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests RNN model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default='../Data/train', help='Training data')
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default='../Data/test', help='Testing data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='../Data/vocab.txt', help='Path to vocabulary file')
    command_parser.set_defaults(func=train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default='../Data/test', help='Training data')
    command_parser.add_argument('-m', '--model-path', help='Training data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='../Data/vocab.txt', help='Path to vocabulary file')
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help='Training data')
    command_parser.set_defaults(func=evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help='Training data')
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default='../Data/vocab.txt', help='Path to vocabulary file')
    command_parser.set_defaults(func=shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)