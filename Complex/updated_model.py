#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import pdb
import logging

import tensorflow as tf
from model import Model
import tqdm
import utils

logger = logging.getLogger('final')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import matplotlib
import matplotlib.pyplot as plt

train_loss = list()
dev_loss = list()

class UpdatedModel(Model):

    def __init__(self, config, report=None):
        self.config = config
        self.report = report


    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError('Each Model must re-implement this method.')


    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError('Each Model must re-implement this method.')


    def evaluate(self, sess, examples):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
        Returns:
            The F1 score for predictions.
        """
        pass


    def run_epoch(self, sess, train_examples, dev_set):
        global train_loss
        total = 0
        seen = 0
        batches = utils.get_batches(train, config.batch_size)
        for batch in tqdm(batches):
            loss = self.train_on_batch(sess, *batch)
            total += loss
            seen += 1
            if self.report:
                self.report.log_train_loss(loss)
        train_loss.append(float(total/seen))
        print("")

        logger.info('Evaluating on development data')
        #_ = self.evaluate(sess, dev_set)
    

        f1 = entity_scores[-1]
        return f1


    def output(self, sess, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        preds = []
        batches = utils.get_batches(train, config.batch_size)

        for batch in tqdm(batches):
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)

        return self.consolidate_predictions(inputs, preds)


    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        global train_loss
        global dev_loss
        best_score = 0.

        train = self.preprocess_sequence_data(train_examples_raw)
        dev = self.preprocess_sequence_data(dev_set_raw)

        epochs = list()
        for epoch in range(self.config.n_epochs):
            epochs.append(epoch + 1)
            logger.info('Epoch %d out of %d', epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train, dev)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info('New best score! Saving model in %s', self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        plt.plot(epochs, train_loss, label='train loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        output_path = 'plot%.2f.png', time.time()
        plt.savefig(output_path)
        
        return best_score