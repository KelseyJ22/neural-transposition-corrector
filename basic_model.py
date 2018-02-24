import tensorflow as tf

class Model:
	def __init__(self, config):
		print 'creating model'
		self.config = config

		# placeholders
		self.encoderInputs  = None
		self.decoderInputs  = None 
		self.decoderTargets = None
		self.decoderWeights = None  

		# ops
		self.loss = None
		self.optimizer_op = None
		self.outputs = None

		# construct
		self.build_network()


	def create_rnn_cell(self):
		cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size,)
		cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prop=1.0, output_keep_prob=self.config.dropout)
		return cell


	def build_network(self):
		cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell()] for _ in range(self.config.num_layers))

		with tf.name_scope('placeholder_encoder'):
			self.encoder_inputs = [tf.placeholder(tf.int32, [None, ]) for _ in range(self.config.encoder_len)]  # batch size * sequence length * input dim

		with tf.name_scope('placeholder_decoder'):
			self.decoder_inputs  = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in range(self.config.decoder_len)]  # same sentence length for input and output
			self.decoder_targets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in range(self.config.decoder_len)]
			self.decoder_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.config.decoder_len)]

		
		decoder_outputs, states = tf.contrib.seq2seq.embedding_rnn_seq2seq(self.encoder_inputs, self.decoder_inputs, cell, self.config.vocab_size, self.config.vocab_size, embedding_size=self.config.embedding_size, feed_previous=bool(self.config.test)) 

		if self.config.test:
			self.outputs = decoder_outputs
		else:
			self.loss = tf.contrib.seq2seq.sequence_loss(decoder_outputs, self.decoder_targets, self.decoder_weights, self.config.vocab_size)
			tf.summary.scalar('loss', self.loss)

		opt = tf.train.AdamOptimizer(learning_rate = self.config.lr)
		self.optimizer_op = opt.minimize(self.loss)


	def train_step(self, batch, training=True):
		feed_dict = {}
		ops = None
		if not training: # TEST
			self.config.test = True
			for i in range(0, self.config.encoder_len):
				feed_dict[self.encoder_inputs[i]] = batch.encoder_seq[i]
			feed_dict[self.decoder_inputs[0]] = [self.data.go] # TODO: what is this?

			ops = (self.outputs,)

		else: # TRAIN
			self.config.test = False
			for i in range(0, self.config.encoder_len):
				feed_dict[self.encoder_inputs[i]] = batch.encoder_seq[i]
			for i in range(0, self.config.decoder_len):
				feed_dict[self.decoder_inputs] = batch.decoder_seq[i]
				feed_dict[self.decoder_targets[i]] = batch.target_seq[i]
				feed_dict[self.decoder_weights[i]] = batch.weights[i]
			ops = (self.optimizer_op, self.loss)

		return ops, feed_dict
