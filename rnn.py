import datetime
import tensorflow as tf
import numpy as np 
import math
from tqdm import tqdm
from basic_model import Model
import utils


class Config:
	def __init__(self):
		self.character_set = 'abcdefghijklmnopqrstuvwxyz' # only dealing with lowercase alphabetic characters
		self.hidden_size = 512
		self.dropout = 0.9
		self.max_len = 10
		self.vocab_size = 2000
		self.lr = 0.001
		self.charset_size = len(character_set)
		self.embedding_size = 3 * charset_size
		self.num_layers = 2
		self.test = False
		self.batch_size = 100


class RNN:
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


	def create_embedding(self, word):
		first = self.character_set.find(word[0])
		last = self.character_set.find(word[-1])
		bow = list()
		for i in range(1, len(word)-1):
			bow.append(self.character_set.find(word[i]))
		embedding = np.zeros((3 * self.charset_size))
		embedding[first] = 1
		for char in bow:
			embedding[self.charset_size + char] = 1
		embedding[self.charset_size * 2 + last] = 1

		return embedding


	def load_embeddings(self, batch):
		embedded = np.array((len(batch), self.max_sentence_len, self.embedding_size))
		for i, sentence in batch:
			for j, word in sentence:
				embedded[i][j] = self.create_embedding(word)

		return embedded


 	def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len, self.embedding_size), name="inputs")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len), name="labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout")


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
        loss = tf.reduce_mean(loss_vector)
        return loss


    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op


    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


	def run(self):
		train, test = utils.load_data('Data/movie_lines.txt')
		embeddings = utils.load_embeddings()

		self.sess.run(tf.global_variables_initializer())
		self.writer.add_graph(sess.graph)
		merged_summaries = tf.summary.merge_all()

		print 'Starting training...'

		for epoch in range(self.num_epochs):
				print '-----Epoch', epoch, '-----'
				batches = utils.get_batches(train, self.config.batch_size)

				start_time = datetime.datetime.now()
				for batch in tqdm(batches):
					embedded = self.load_embeddings(batch)
					loss = self.train_on_batch(sess, embedded)
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
					if self.global_step % self.save_interval == 0:
						self.save_session(sess)

				end_time = datetime.datetime.now()
				print 'Epoch finish in ', end_time-start_time, 'ms'


	def save_session(self, sess):
		print 'Saving session at checkpoint', self.global_step
		name = self.model_dir + str(self.global_step)
		self.saver.save(sess, name)
		print 'Save complete with name', name