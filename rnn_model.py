"""
- construct word-level seq2seq autoencoder with input = output
	Fundamental problem = misspellings are OOV
- construct character-level model and backprop into character vectors to learn them
- create character-based word vectors using weighted average of current character and its neighbors as a representation of each character
- idea = use a mini-RNN (or CNN??) to learn the input to the primary autoencoder
"""
import datetime
import tensorflow as tf
import numpy as np 
import math
from tqdm import tqdm
from basic_model import model # TODO implement

class OuterRNN:
	def __init__(self):
		self.model = model()
        self.writer = tf.summary.FileWriter(self._getSummaryName())
        self.saver = tf.train.Saver(max_to_keep=200)
		self.global_step = 0
		self.sess = tf.Session() # TODO: any params?
		self.num_epochs = 100 # TODO: tune


	def main(self):
		self.sess.run(tf.global_variables_initializer())
		self.loadEmbeddings(self.sess)
		self.writer.add_graph(sess.graph)
		mergedSummaries = tf.summary.merge_all() # TODO: what does this do?

		print 'Starting training...'

		for epoch in range(self.num_epochs):
				print '-----Epoch', epoch, '-----'
				batches = utils.get_batches() # TODO: write this

				start_time = datetime.datetime.now()
				for batch in tqdm(batches):
					ops, feed = self.model.train_step(batch)
					_, loss, summary = sess.run(ops + (mergedSummaries,), feed)
					self.writer.add_summary(summary, self.global_step)
                    self.global_step += 1

                    # training status
                    if self.global_step % 100 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                        tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.global_step, loss, perplexity))

                    # save checkpoint
                    if self.global_step % 100 == 0:
                        self._saveSession(sess)


                end_time = datetime.datetime.now()
                print 'Epoch finish in ', end_time-start_time, 'ms'