import numpy as np
import math
import random
import re
import tensorflow as tf
import pickle

pad = '0'
max_length = 10
unknown = '<UNK>'
characters = 'abcdefghijklmnopqrstuvwxyz'

def create_weighted_embedding(word):
	max_word_len = 8
	characters = 'abcdefghijklmnopqrstuvwxyz'
	charset_size = len(characters)
	word_vec = list()

	for i in range(0, max_word_len):
	    if i < len(word):
	        ind = characters.find(word[i])
	        new_vec = [0] * charset_size
	        new_vec[ind] = 1

	        if i-1 >= 0:
	            ind = characters.find(word[i-1])
	            new_vec[ind] = 0.5
	        elif i+1 < len(word):
	            ind = characters.find(word[i+1])
	            new_vec[ind] = 0.5
	        elif i-2 >= 0:
	            ind = characters.find(word[i-2])
	            new_vec[ind] = 0.06
	        elif i+2 < len(word):
	            ind = characters.find(word[i+2])
	            new_vec[ind] = 0.06

	        word_vec += new_vec
	    else:
	        word_vec += [0] * charset_size # padding so all word vectors are the same size
	return word_vec


def create_embedding(word):
	character_set = 'abcdefghijklmnopqrstuvwxyz'
	charset_size = len(character_set)

	first = character_set.find(word[0])
	last = character_set.find(word[-1])
	bow = list()
	for i in range(1, len(word)-1):
		bow.append(character_set.find(word[i]))
	embedding = np.zeros((3 * charset_size))
	embedding[first] = 1
	for char in bow:
		embedding[charset_size + char] = 1
	embedding[charset_size * 2 + last] = 1
	return embedding


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

		parsed.append(sent)
	return parsed


def vectorize(input_list):
	listified = input_list.split(' ')
	res = np.zeros([len(listified)-1]) # last element is '\n' and we don't want it
	for i in range(0, len(listified)-1):
		res[i] = float(listified[i])
	return res


def shuffle_string(string):
    chars = list(string)
    random.shuffle(chars)
    return ''.join(chars)


def add_error(word):
	#inner_chars = word[1:-1]
	#new_word = word[0] + shuffle_string(inner_chars) + word[-1]
	new_word = shuffle_string(word)
	new_char = random.randint(0, 26-1)
	loc = random.randint(0, len(word))
	new_word = new_word[0:loc-1] + characters[new_char] + new_word[loc:]
	return new_word


def divide(string):
	chunks = list()
	i = 3
	prev_i = 0
	while i < len(string):
		chunks.append(string[prev_i:i])
		prev_i = i 
		i += 3
	chunks.append(string[prev_i:]) # otherwise will miss end of the string
	return chunks


def add_local_error(word):
	inner_chars = word[1:-1]
	if len(inner_chars) > 3:
		inners = divide(inner_chars)
		new_word = word[0]
		for chunk in inners:
			new_word += shuffle_string(chunk)
		new_word += word[-1]
	else:
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
	word = word.strip().lower()
	word = word.replace(',', '')
	word = word.replace(';', '')
	word = word.replace(':', '')
	word = word.replace(')', '')
	word = word.replace('(', '')
	word = word.replace(']', '')
	word = word.replace('[', '')
	word = word.replace('-', '')
	word = word.replace('<u>', '')
	word = word.replace("\"", '')
	return str(word)


def generate_errors(data, frequencies):
	errored = list()
	for sentence in data:
		cleaned_sentence = list()
		new_sentence = list()
		for word in sentence:
			word = clean(word)
			if len(word) > 0:
				if word in frequencies and frequencies[word] > 80:
					cleaned_sentence.append(word)
					if is_transposable(word):
						gen_error = 0 #random.randint(0, 3)
						if gen_error == 0:
							new_sentence.append(add_error(word))
						else:
							new_sentence.append(word)
					else:
						new_sentence.append(word)
				else:
					cleaned_sentence.append(unknown)
					new_sentence.append(unknown)

		errored.append((new_sentence[:10], cleaned_sentence[:10])) # input, output pair
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
				w = clean(word)
				if len(w) > 0:
					if w in frequencies:
						frequencies[w] += 1
					else:
						frequencies[w] = 1

	print('read in ', len(dataset), 'samples')
	print('found a total of', len(frequencies), 'words')

	# add transposition errors
	parsed_data = generate_errors(dataset, frequencies)

	# train/test split
	train, test = parsed_data[0:1000000], parsed_data[1000000:1010000]
	
	return train, test, frequencies


def pad_sequences(data):
	ret = []

	for sentence, labels in data:
		new_sentence = list()
		new_labels = list()
		mask = list()

		i = 0
		while len(new_sentence) < max_length:
			if i < len(sentence): # still in the sentence
				new_sentence.append(sentence[i])
				new_labels.append(labels[i])                
				mask.append(True)
			else: # pad with zeros
				new_sentence.append('0')
				new_labels.append(2001)
				mask.append(False)
			i += 1

		ret.append((new_sentence, new_labels, mask))

	return ret


def batchify(data, batch_size):
	inputs = list()
	labels = list()
	masks = list()
	for i in range(0, len(data)):
		sample = data[i]
		inputs.append(list(sample[0]))
		labels.append(sample[1])
		masks.append(sample[2])
	return (inputs, labels, masks)


def get_batch(data, batch_size):
	for i in range(0, batch_size):
		yield data[i:i + batch_size] 


def get_batches(data, batch_size):
	batches = list()
	random.shuffle(data) # reshuffle for each iteration
	for batch in get_batch(data, batch_size):
		batches.append(batchify(batch, batch_size))

	return batches


def clean_sentence(sentence):
	res = list()
	for word in sentence:
		res.append(word.strip())
	return res


def save(path, map1, map2):
	if not os.path.exists(path):
		os.makedirs(path)
	with open(os.path.join(path, '.pkl'), 'w') as f:
		pickle.dump([map1, map2], f)


def parse_str(line):
	split = line.split('|')
	i = split[0]
	o = split[1]
	inp = clean_sentence(i.split(','))
	outp = clean_sentence(o.split(','))

	return (inp, outp)


def load_from_file():
	test_file = 'Data/test'
	test_file_obj = open(test_file, 'r')
	test = pickle.load(test_file_obj)
	test_file_obj.close()

	train_file = 'Data/train'
	train_file_obj = open(train_file, 'r')
	train = pickle.load(train_file_obj)
	train_file_obj.close()

	word2id_file = 'Data/word2id'
	word2id_file_obj = open(word2id_file, 'r')
	word_to_id = pickle.load(word2id_file_obj)
	word2id_file_obj.close()

	id_to_word = dict()
	for word in word_to_id:
		id_to_word[word_to_id[word]] = word

	return train, test, word_to_id, id_to_word


def GRU(inputs, state, input_size, state_size):
	with tf.variable_scope('GRU'):
		W_r = tf.get_variable('W_r', shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		U_r = tf.get_variable('U_r', shape=[input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		b_r = tf.get_variable('b_r', shape=[state_size,], initializer = tf.constant_initializer(0.0))


		W_z = tf.get_variable('W_z', shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		U_z = tf.get_variable('U_z', shape=[input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		b_z = tf.get_variable('b_z', shape=[state_size,], initializer = tf.constant_initializer(0.0))

		W_o = tf.get_variable('W_o', shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		U_o = tf.get_variable('U_o', shape=[input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
		b_o = tf.get_variable('b_o', shape=[state_size,], initializer = tf.constant_initializer(0.0))

		z_t = tf.nn.sigmoid(tf.matmul(inputs, U_z) + tf.matmul(state, W_z) + b_z)
		r_t = tf.nn.sigmoid(tf.matmul(inputs, U_r) + tf.matmul(state, W_r) + b_r)
		o_t = tf.nn.tanh(tf.matmul(inputs, U_o) + tf.matmul(r_t * state, W_o)+ b_o)
		new_state = z_t * state + (1 - z_t) * o_t

	output = new_state
	return output, new_state