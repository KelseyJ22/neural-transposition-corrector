#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: A window into NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import time
import logging
from datetime import datetime

import tensorflow as tf
from utils.model import Model

from utils.util import print_sentence, write_conll, ConfusionMatrix, Progbar, minibatches, write_ngrams
from utils.data_util import load_and_preprocess_data, load_embeddings, read_ngrams, ModelHelper, get_chunks
from utils.defs import LBLS
from utils.model import Model
from utils.rnn_cell import RNNCell
from utils.gru_cell import GRUCell


logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """
    n_word_features = 2 # Number of features for every word in the input.
    n_gram_size = 5
    n_window_features = n_word_features * n_gram_size # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = len(LBLS)
    dropout = 0.5
    embed_size = 50
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001
    hidden_size = 300

    def __init__(self, output_path=None):
        self.cell = 'rnn'

        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/window/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_book_predictions.conll"


class NGramModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_window_features), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch

        return feed_dict


    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
        L = tf.Variable(self.pretrained_embeddings)
        lookups = tf.nn.embedding_lookup(L, self.input_placeholder)
        embeddings = tf.reshape(lookups, [-1, self.config.max_length, self.config.n_window_features*self.config.embed_size])

        return embeddings


    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = list() # Predicted output at each timestep should go here!

        if self.config.cell == "rnn":
            cell = RNNCell(Config.n_window_features * Config.embed_size, Config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(Config.n_window_features * Config.embed_size, Config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        U = tf.get_variable("U", shape=[self.config.hidden_size, self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", initializer=tf.zeros([self.config.n_classes,]))
        h_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size])
        
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.max_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                o_t, h_t = cell(x[:,time_step,:], h_t) # updates h_t for next iteration
                o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
                y_t = tf.matmul(o_drop_t, U) + b2
                preds.append(y_t)

        preds = tf.transpose(tf.pack(preds), perm=[1,0,2])
        preds = tf.reshape(preds, [-1, self.config.max_length])
        return preds


    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        print pred.get_shape()
        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
        loss = tf.reduce_mean(loss_vector)
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
        return examples


    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        i = 0
        for sentence, label in examples_raw:
            label_ = preds[i]
            i += 1
            ret.append([sentence, label, label_])
        return ret


    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=LBLS)
        # Double check how p, r are calculated
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, label, label_  in self.output(sess, examples_raw, examples):
            token_cm.update(label, label_)
            correct_preds += label != label_
            total_preds += 1
            total_correct += label != label_

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)


    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")

        logger.info("Evaluating on development data")
        token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1


    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)


    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score


    def __init__(self, helper, config, pretrained_embeddings, report=None):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.helper = helper
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.report = None

        self.build()


def do_train(args):
    # Set up some parameters.
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = NGramModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [preds for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_ngrams(f, output)


def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_ngrams(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = NGramModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)
