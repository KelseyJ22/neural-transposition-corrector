import datetime
import tensorflow as tf
import numpy as np 
import math
from tqdm import tqdm
import utils


class Config:
	def __init__(self):
		self.character_set = 'abcdefghijklmnopqrstuvwxyz' # only dealing with lowercase alphabetic characters
		self.hidden_size = 512
		self.dropout = 0.9
		self.max_sentence_len = 10
		self.vocab_size = 2000
		self.lr = 0.001
		self.charset_size = len(self.character_set)
		self.embedding_size = 3 * self.charset_size
		self.num_layers = 2
		self.test = False
		self.batch_size = 100


class RNN:
	def __init__(self):
		self.config = Config()
		self.model_dir = 'Psych209_RNN'
		self.global_step = 0
		self.num_epochs = 100
		self.save_interval = 1000
		self.print_interval = 100
		self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_len, self.config.embedding_size), name="inputs")
		self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_len), name="labels")
		self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


	def create_embedding(self, word):
		first = self.config.character_set.find(word[0])
		last = self.config.character_set.find(word[-1])
		bow = list()
		for i in range(1, len(word)-1):
			bow.append(self.config.character_set.find(word[i]))
		embedding = np.zeros((3 * self.config.charset_size))
		embedding[first] = 1
		for char in bow:
			embedding[self.config.charset_size + char] = 1
		embedding[self.config.charset_size * 2 + last] = 1

		return embedding


	def load_embeddings(self, inputs, labels):
		#embedded = np.array((len(inputs), self.config.max_sentence_len, self.config.embedding_size))
		#labels = np.array((len(labels), self.config.max_sentence_len))
		embedded = list()
		labels = list()
		for i in range(0, len(inputs)):
			sentence = inputs[i]
			embedded.append(list())
			for j in  range(0, len(sentence)):
				word = sentence[j]
				embedded[i].append(list())
				embedded[i][j] = self.create_embedding(word)

		for i in range(0, len(labels)):
			sentence = labels[i]
			labels.append(list())
			for j in  range(0, len(sentence)):
				word = sentence[j]
				labels[i].append(list())
				labels[i][j] = self.word_to_id[word]

		return np.asarray(embedded), np.asarray(labels)


	"""def add_placeholders(self):
		self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_len, self.config.embedding_size), name="inputs")
		self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_sentence_len), name="labels")
		self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")"""


	def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
		feed_dict = {self.dropout_placeholder: dropout}
		if labels_batch is not None:
		    feed_dict[self.labels_placeholder] = labels_batch
		if inputs_batch is not None:
		    feed_dict[self.input_placeholder] = inputs_batch

		return feed_dict


	def add_prediction_op(self):
		data = self.input_placeholder
		dropout_rate = self.dropout_placeholder

		preds = list() # predicted output at each timestep

		cell = LSTMCell(config.embedding_size, config.hidden_size)

		U = tf.get_variable('U', shape=[self.config.hidden_size, self.config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
		b2 = tf.get_variable('b2', initializer=tf.zeros([self.config.vocab_size,]))
		h_t = tf.zeros([tf.shape(x)[0], self.config.hidden_size])

		with tf.variable_scope('RNN'):
		    for time_step in range(self.config.max_sentence_len):
		        if time_step > 0:
		            tf.get_variable_scope().reuse_variables()

		        o_t, h_t = cell(data[:,time_step,:], h_t) # updates h_t for next iteration
		        o_drop_t = tf.nn.dropout(o_t, self.dropout_placeholder)
		        y_t = tf.matmul(o_drop_t, U) + b2
		        preds.append(y_t)

		preds = tf.transpose(tf.pack(preds), perm=[1,0,2]) # TODO check this
		preds = tf.reshape(preds, [-1, self.config.max_sentence_len])
		return preds


	def add_loss_op(self, pred):
		loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred)
		self.loss = tf.reduce_mean(loss_vector)
		return loss


	def add_training_op(self, loss):
		self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
		return train_op


	def predict_on_batch(self, sess, inputs_batch):
		feed = self.create_feed_dict(inputs_batch)
		predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
		return predictions


	def train_on_batch(self, sess, inputs_batch, labels_batch):
		feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
		return loss


	def load_word_lookup(self, frequencies):
		lookups = dict()
		current_count = 0
		for word in frequencies:
			if frequencies[word] > 80:
				if word not in lookups:
					lookups[word] = current_count
					current_count += 1

		return lookups


	#def run(self, sess, saver, writer, train, test, frequencies):
	def run(self, sess, writer, train, test, frequencies):
		self.word_to_id = self.load_word_lookup(frequencies)
		for epoch in range(self.num_epochs):
			print '-----Epoch', epoch, '-----'
			batches = utils.get_batches(train, self.config.batch_size)

			start_time = datetime.datetime.now()
			for batch in tqdm(batches):
				embedded, labels = self.load_embeddings(batch[0], batch[1])
				loss = self.train_on_batch(sess, embedded, labels)
				summary = tf.summary.scalar('loss', loss) # for logging
				self.writer.add_summary(summary, self.global_step)
				self.global_step += 1

				# training status
				if self.global_step % self.print_interval == 0:
					perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
					tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (self.global_step, loss, perplexity))
					# run test periodically
					loss = self.predict_on_batch(sess, embedded)

				# save checkpoint
				#if self.global_step % self.save_interval == 0:
				#	self.save_session(sess)

			end_time = datetime.datetime.now()
			print 'Epoch finished in ', end_time-start_time, 'ms'


	def save_session(self, sess):
		print 'Saving session at checkpoint', self.global_step
		name = self.model_dir + str(self.global_step)
		self.saver.save(sess, name)
		print 'Save complete with name', name


def run_rnn():
		train, test, frequencies = utils.load_data('Data/movie_lines.txt')
		embeddings = utils.load_embeddings()
		
		with tf.Graph().as_default():
			model = RNN()
			writer = tf.summary.FileWriter(model.model_dir)
			merged_summaries = tf.summary.merge_all()

			print 'Starting training...'
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				#saver = tf.train.Saver()
				writer.add_graph(sess.graph)
				#model.run(sess, saver, writer, train, test, frequencies)
				model.run(sess, writer, train, test, frequencies)
				

run_rnn()