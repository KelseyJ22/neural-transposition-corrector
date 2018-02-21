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
from basic_model import Model
import utils

class Config:
	def __init__(self):
		self.hidden_size = 512
		self.dropout = 0.9
		self.encoder_len = 10 # maximum length of the sentence
		self.decoder_len = 12
		self.vocab_size = 20000 # TODO check
		self.lr = 0.001
		self.test = False
		self.embedding_size = 100
		self.num_layers = 2
		self.test = False


class OuterRNN:
	def __init__(self):
		self.config = Config()
		self.model = Model(self.config)
		self.model_dir = 'Psych209_RNN'
		self.writer = tf.summary.FileWriter(self.model_dir)
		self.saver = tf.train.Saver(max_to_keep=200)
		self.global_step = 0
		self.sess = tf.Session()
		self.num_epochs = 100
		self.save_interval = 1000
		self.print_interval = 100


	def run(self):
		train, test = utils.load_data('Data/movie_lines.txt')
		embeddings = utils.load_embeddings()

		self.sess.run(tf.global_variables_initializer())
		self.writer.add_graph(sess.graph)
		merged_summaries = tf.summary.merge_all()

		print 'Starting training...'

		for epoch in range(self.num_epochs):
				print '-----Epoch', epoch, '-----'
				batches = utils.get_batches()

				start_time = datetime.datetime.now()
				for batch in tqdm(batches):
					ops, feed = self.model.train_step(batch, training=True)
					_, loss, summary = sess.run(ops + (merged_summaries,), feed)
					self.writer.add_summary(summary, self.global_step)
					self.global_step += 1

					# training status
					if self.global_step % self.print_interval == 0:
						perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
						tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.global_step, loss, perplexity))
						# run test periodically
						ops, feed = self.model.train_step(batch, training=False)
						_, loss, summary = sess.run(ops + (merged_summaries,), feed)

					# save checkpoint
					if self.global_step % self.save_interval == 0:
						self.save_session(sess)

				end_time = datetime.datetime.now()
				print 'Epoch finish in ', end_time-start_time, 'ms'

	def save_session(self, sess):
		print 'Saving session at checkpoint', self.global_step
		name = self.model_dir + str(self.global_step)
		self.saver.save(sess, name)
		print 'Save complete with name', name

"""rnn_model = OuterRNN()
rnn_model.run()"""
#train, test = utils.load_data('Data/movie_lines.txt')
embeddings = utils.load_embeddings()