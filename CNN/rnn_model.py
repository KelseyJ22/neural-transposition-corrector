#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import pdb
import logging

import tensorflow as tf
from model import Model
from tqdm import tqdm
import utils
import time

logger = logging.getLogger('final')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import matplotlib
import matplotlib.pyplot as plt

train_loss = list()
dev_loss = list()

class RNNModel(Model):

    def __init__(self, config):
        self.config = config


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
        batches = utils.get_batches(train_examples, self.config.batch_size)
        for batch in tqdm(batches):
            loss = self.train_on_batch(sess, *batch)
            total += loss
            seen += 1
        
        train_loss.append(float(total/seen))
    
        return train_loss[-1]


    def output(self, sess, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        preds = list()
        batched_in = list()
        batched_mask = list()
        inputs = self.preprocess_sequence_data(inputs)

        batches = utils.get_batches(inputs, self.config.batch_size)

        for batch in tqdm(batches):
            batched_in += list(batch[0])
            batched_mask += list(batch[2])
            preds_ = self.predict_on_batch(sess, batch[0], batch[2]) # only pass inputs and masks
            preds += list(preds_)

        return self.consolidate_predictions(batched_in, batched_mask, preds)


    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        global train_loss
        global dev_loss
        best_score = 1000

        train = self.preprocess_sequence_data(train_examples_raw)
        dev = self.preprocess_sequence_data(dev_set_raw)

        epochs = list()
        for epoch in range(self.config.n_epochs):
            epochs.append(epoch + 1)
            logger.info('Epoch %d out of %d', epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train, dev)
            print score
            if score < best_score:
                best_score = score
                if saver:
                    logger.info('New best score! Saving model in %s', self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            

        plt.plot(epochs, train_loss, label='train loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        output_path = 'plots/plot' + str(time.time()) + '.png'
        plt.savefig(output_path)
        
        return best_score