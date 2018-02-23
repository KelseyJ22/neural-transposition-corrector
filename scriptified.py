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
		self.dropout = 0.9
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


def load_embeddings(inputs, labels):
	embedded = list()
	labels = list()
	for i in range(0, len(inputs)):
		sentence = inputs[i]
		embedded.append(list())
		for j in  range(0, len(sentence)):
			word = sentence[j]
			embedded[i].append(list())
			embedded[i][j] = create_embedding(word)

	for i in range(0, len(labels)):
		sentence = labels[i]
		labels.append(list())
		for j in  range(0, len(sentence)):
			word = sentence[j]
			labels[i].append(list())
			labels[i][j] = word_to_id[word]

	return np.asarray(embedded), np.asarray(labels)


def load_word_lookup(frequencies):
	lookups = dict()
	current_count = 0
	for word in frequencies:
		if frequencies[word] > 80:
			if word not in lookups:
				lookups[word] = current_count
				current_count += 1

	return lookups


config = Config()

data = tf.placeholder(tf.int32, shape=(None, config.max_sentence_len, config.embedding_size), name="inputs")
labels = tf.placeholder(tf.int32, shape=(None, config.max_sentence_len), name="labels")
dropout_rate = tf.placeholder(tf.float32, name="dropout")

preds = list() # will be predicted output at each timestep
cell = LSTMCell(config.embedding_size, config.hidden_size)

U = tf.get_variable('U', shape=[config.hidden_size, config.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2', initializer=tf.zeros([config.vocab_size,]))
h_t = tf.zeros([tf.shape(x)[0], config.hidden_size])

with tf.variable_scope('RNN'):
    for time_step in range(config.max_sentence_len):
        if time_step > 0:
            tf.get_variable_scope().reuse_variables()

        o_t, h_t = cell(data[:,time_step,:], h_t) # updates h_t for next iteration
        o_drop_t = tf.nn.dropout(o_t, dropout_placeholder)
        y_t = tf.matmul(o_drop_t, U) + b2
        preds.append(y_t)

preds = tf.transpose(tf.pack(preds), perm=[1,0,2]) # TODO check this
pred = tf.reshape(preds, [-1, config.max_sentence_len])

loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=pred)
loss = tf.reduce_mean(loss_vector)

opt = tf.train.AdamOptimizer(config.lr).minimize(loss)

train, test, frequencies = utils.load_data('Data/movie_lines.txt')
embeddings = utils.load_embeddings()

with tf.Graph().as_default():
	writer = tf.summary.FileWriter(model.model_dir)
	merged_summaries = tf.summary.merge_all()

	print 'Starting training...'
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(sess)
		writer.add_graph(sess.graph)
		word_to_id = load_word_lookup(frequencies)

		for epoch in range(config.num_epochs):
			print '-----Epoch', epoch, '-----'
			batches = utils.get_batches(train, config.batch_size)

			start_time = datetime.datetime.now()
			for batch in tqdm(batches):

				embedded, one_hot_labels = load_embeddings(batch[0], batch[1])
				feed = {data: embedded, labels: one_hot_labels, dropout: config.dropout}
				_, loss = sess.run([opt, loss], feed_dict=feed)

				summary = tf.summary.scalar('loss', loss) # for logging
				writer.add_summary(summary, config.global_step)
				config.global_step += 1

				# training status
				if config.global_step % config.print_interval == 0:
					perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
					tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (config.global_step, loss, perplexity))
					# run test periodically
					#feed = create_feed_dict(embedded)
					#predictions = sess.run(tf.argmax(pred, axis=1), feed_dict=feed)

				# save checkpoint
				if config.global_step % config.save_interval == 0:
					print 'Saving session at checkpoint', config.global_step
					name = config.model_dir + str(config.global_step)
					saver.save(sess, name)
					print 'Save complete with name', name


			end_time = datetime.datetime.now()
			print 'Epoch finished in ', end_time - start_time, 'ms'
				




