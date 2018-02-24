import datetime
import tensorflow as tf
import numpy as np 
import math
from tqdm import tqdm
import utils
from model import Model


class Config:
	def __init__(self):
		self.character_set = 'abcdefghijklmnopqrstuvwxyz' # only dealing with lowercase alphabetic characters
		self.hidden_size = 512
		self.max_sentence_len = 10
		self.vocab_size = 2000
		self.lr = 0.001
		self.charset_size = len(self.character_set)
		self.embedding_size = 3 * self.charset_size
		self.num_layers = 2
		self.test = False
		self.batch_size = 100
		self.model_dir = 'Psych209_RNN'
		self.global_step = 0
		self.num_epochs = 100
		self.save_interval = 1000
		self.print_interval = 100
		self.keep_prob = 0.9


def create_embedding(word):
	first = config.character_set.find(word[0])
	last = config.character_set.find(word[-1])
	bow = list()
	for i in range(1, len(word)-1):
		bow.append(config.character_set.find(word[i]))
	embedding = np.zeros((3 * config.charset_size))
	embedding[first] = 1
	for char in bow:
		embedding[config.charset_size + char] = 1
	embedding[config.charset_size * 2 + last] = 1

	return embedding


def load_embeddings(config, inputs, labels):
	embedded = list()
	parsed_labels = list()
	for i in range(0, len(inputs)):
	# embedded will be of shape [batch_size, max_sentence_length, embedding_size]
		sentence = inputs[i]
		embedded.append(list())
		for j in  range(0, len(sentence)):
			word = sentence[j]
			embedded[i].append(list())
			embedded[i][j] = create_embedding(word)

	parsed_labels = np.zeros((config.batch_size, config.max_sentence_len))
	for i in range(0, config.batch_size):
		sentence = labels[i]
		for j in  range(0, config.max_sentence_len):
			word = sentence[j]
			parsed_labels[i][j] = word_to_id[word]

	return np.asarray(embedded), np.asarray(parsed_labels)


def load_word_lookup(frequencies):
	lookups = dict()
	current_count = 1
	for word in frequencies:
		if frequencies[word] > 80:
			if word not in lookups:
				lookups[word] = current_count
				current_count += 1

	# map all non-word characters to 0
	lookups['<s>'] = 0
	lookups['</s>'] = 0
	lookups['<UNK>'] = 0
	lookups['0'] = 0
	return lookups

with tf.Graph().as_default():

	config = Config()

	data = tf.placeholder(tf.float32, shape=(None, config.max_sentence_len, config.embedding_size), name='data')
	labels = tf.placeholder(tf.int32, shape=(None, config.max_sentence_len), name='labels')


	U = tf.get_variable("U", shape=[config.hidden_size, config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", shape=[config.vocab_size,], initializer = tf.constant_initializer(0))
	h_t = tf.zeros([tf.shape(data)[0], config.hidden_size]) # initialize hidden state

	preds = list()
	with tf.variable_scope("RNN"):
		for time_step in range(config.max_sentence_len):
			if time_step > 0:
				tf.get_variable_scope().reuse_variables()

			o_t, h_t = utils.GRU(data[:,time_step,:], h_t, config.embedding_size, config.hidden_size)
			o_drop_t = tf.nn.dropout(o_t, config.keep_prob)
			y_t = tf.matmul(o_drop_t, U) + b2
			preds.append(y_t)

	loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=preds)
	loss = tf.reduce_mean(loss_vector)

	opt = tf.train.AdamOptimizer(config.lr).minimize(loss)

	train, test, frequencies = utils.load_data('Data/movie_lines.txt')
	embeddings = utils.load_embeddings()

	writer = tf.summary.FileWriter(config.model_dir)
	merged_summaries = tf.summary.merge_all()

	print 'Starting training...'
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer.add_graph(sess.graph)
		word_to_id = load_word_lookup(frequencies)

		for epoch in range(config.num_epochs):
			print '-----Epoch', epoch, '-----'
			batches = utils.get_batches(train, config.batch_size)

			start_time = datetime.datetime.now()
			for batch in tqdm(batches):

				embedded, one_hot_labels = load_embeddings(config, batch[0], batch[1])
				feed = {data: embedded, labels: one_hot_labels}
				_, curr_loss = sess.run((opt, loss), feed_dict=feed)

				#summary = tf.summary.scalar('loss', loss) # for logging
				#writer.add_summary(summary, config.global_step)
				config.global_step += 1

				# training status
				if config.global_step % config.print_interval == 0:
					perplexity = math.exp(float(curr_loss)) if curr_loss < 300 else float('inf')
					tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (config.global_step, curr_loss, perplexity))
					# run test periodically
					#feed = create_feed_dict(embedded)
					#predictions = sess.run(tf.argmax(pred, axis=1), feed_dict=feed)

				# save checkpoint
				if config.global_step % config.save_interval == 0:
					print 'Saving session at checkpoint', config.global_step
					name = config.model_dir + str(config.global_step)
					saver = tf.train.Saver()
					saver.save(sess, name)
					print 'Save complete with name', name


			end_time = datetime.datetime.now()
			print 'Epoch finished in ', end_time - start_time, 'ms'
				




