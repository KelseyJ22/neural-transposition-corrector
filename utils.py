import numpy as np
import math
import random
import re
import tensorflow as tf

start = '<s>'
end = '</s>'
pad = '0'
max_length = 10
unknown = '<UNK>'


def parse(line):
	parsed = list()
	start = line.rfind('+')
	if start != -1:
		line = line[start+1:]
	sentences = re.split('\.*\?*\!*', line) # split into sentences
	for sentence in sentences: # want each sentence to be its own dataset
		sent = list()
		words = sentence.split(' ')
		for word in words:
			sent.append(word.strip().lower())

		if len(sent) > max_length: # don't want super long sentences
			sent = sent[:max_length]
		parsed.append(sent)
	return parsed


def vectorize(input_list):
	listified = input_list.split(' ')
	res = np.zeros([len(listified)-1]) # last element is '\n' and we don't want it
	for i in range(0, len(listified)-1):
		res[i] = float(listified[i])
	return res


def load_embeddings():
	vecs = open('Data/wordVectors.txt', 'r').readlines()
	vocab = open('Data/vocab.txt', 'r').readlines()
	assert len(vecs) == len(vocab)
	print 'read in ', len(vocab), 'words and wordvectors'
	all_vocab = dict()
	for i in range(0, len(vocab)):
		all_vocab[vocab[i]] = vectorize(vecs[i])
	return all_vocab


def shuffle_string(string):
    chars = list(string)
    random.shuffle(chars)
    return ''.join(chars)


def add_error(word):
	inner_chars = word[1:-1]
	new_word = word[0] + shuffle_string(inner_chars) + word[-1]
	return new_word


def true_len(word):
	count = 0
	for char in word:
		if char.isalpha():
			count += 1
	return count


def is_transposable(word):
	if true_len(word) > 3:
		if true_len(word) < 8:
			if word.isalpha():
				return True
	return False


def clean(word):
	word = word.replace(',', '')
	word = word.replace(';', '')
	word = word.replace(':', '')
	word = word.replace(')', '')
	word = word.replace('(', '')
	word = word.replace(']', '')
	word = word.replace('[', '')
	word = word.replace('-', '')
	return word


def generate_errors(data, frequencies):
	errored = list()
	for sentence in data:
		cleaned_sentence = list()
		new_sentence = list()
		for word in sentence:
			if len(word) > 0:
				if word in frequencies and frequencies[word] > 80:
					cleaned_sentence.append(word)
				else:
					cleaned_sentence.append(unknown)
			if is_transposable(word):
				gen_error = random.randint(0, 3)
				if gen_error == 0:
					word = add_error(word)
			word = clean(word)
			if len(word) > 0:
				if word in frequencies and frequencies[word] > 80:
					new_sentence.append(word)
				else:
					new_sentence.append(unknown)

		if len(new_sentence) > 0:
			errored.append((new_sentence, cleaned_sentence)) # input, output pair
	return errored


def load_data(fname):
	frequencies = dict()
	# read in data
	data = open(fname, 'r')
	dataset = list()
	for line in data.readlines():
		text_data = parse(line)
		for sentence in text_data:
			dataset.append(sentence)

			for word in sentence:
				w = clean(word.strip().lower())
				if w in frequencies:
					frequencies[w] += 1
				else:
					frequencies[w] = 1

	print 'read in ', len(dataset), 'samples'
	print 'found a total of', len(frequencies), 'words'

	# add transposition errors
	parsed_data = generate_errors(dataset, frequencies)

	# train/test split
	train, test = parsed_data[0:100000], parsed_data[100000:110000]
	
	return train, test, frequencies


def batchify(data, batch_size):
	inputs = list()
	labels = list()
	for i in range(0, len(data)):
		sample = data[i]
		inputs.append(list(reversed(sample[0])))
		labels.append([start] + sample[1] + [end])

		inputs[i] = [pad] * (max_length - len(inputs[i])) + inputs[i] # left padding
		labels[i] = labels[i] + [pad] * (max_length - len(labels[i])) # right padding
	return (inputs, labels)


def get_batch(data, batch_size):
	for i in range(0, batch_size):
		yield data[i:i + batch_size] 


def get_batches(data, batch_size):
	batches = list()
	random.shuffle(data) # reshuffle for each iteration
	for batch in get_batch(data, batch_size):
		batches.append(batchify(batch, batch_size))

	return batches


def GRU(inputs, state, input_size, state_size):
	with tf.variable_scope('GRU'):
		W_r = tf.get_variable("W_r", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		U_r = tf.get_variable("U_r", shape=[input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		b_r = tf.get_variable("b_r", shape=[state_size,], initializer = tf.constant_initializer(0.0))


		W_z = tf.get_variable("W_z", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		U_z = tf.get_variable("U_z", shape=[input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		b_z = tf.get_variable("b_z", shape=[state_size,], initializer = tf.constant_initializer(0.0))

		W_o = tf.get_variable("W_o", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		U_o = tf.get_variable("U_o", shape=[input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		b_o = tf.get_variable("b_o", shape=[state_size,], initializer = tf.constant_initializer(0.0))

		z_t = tf.nn.sigmoid(tf.matmul(inputs, U_z) + tf.matmul(state, W_z) + b_z)
		r_t = tf.nn.sigmoid(tf.matmul(inputs, U_r) + tf.matmul(state, W_r) + b_r)
		o_t = tf.nn.tanh(tf.matmul(inputs, U_o) + tf.matmul(r_t * state, W_o)+ b_o)
		new_state = z_t * state + (1 - z_t) * o_t

	output = new_state
	return output, new_state