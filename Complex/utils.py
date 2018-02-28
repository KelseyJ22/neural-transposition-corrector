import numpy as np
import math
import random
import re
import tensorflow as tf
import pickle
import os

pad = '0'
max_length = 10
unknown = '<UNK>'


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
						gen_error = random.randint(0, 3)
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
	train, test = parsed_data[0:100000], parsed_data[100000:110000]

	
	return train, test, frequencies


def pad_sequences(x, y):
	assert x.shape == y.shape

	ret = list()
	for i in range(0, x.shape[0]):
		sentence = x[i]
		labels = y[i]

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
				new_sentence.append(0)
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


def save_results(f, results):
	for i in range(0, len(results)):
		for j in range(0, len(results[i])):
			f.write(str(results[i][j]))
			f.write('\n')


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
	test_file = '../Data/test'
	test_file_obj = open(test_file, 'r')
	test = pickle.load(test_file_obj)
	test_file_obj.close()

	train_file = '../Data/train'
	train_file_obj = open(train_file, 'r')
	train = pickle.load(train_file_obj)
	train_file_obj.close()

	word2id_file = '../Data/word2id'
	word2id_file_obj = open(word2id_file, 'r')
	word_to_id = pickle.load(word2id_file_obj)
	word2id_file_obj.close()

	embeddings_temp = dict()
	id_to_word = dict()
	for word in word_to_id:
		id_to_word[word_to_id[word]] = word
		embeddings_temp[word_to_id[word]] = create_embedding(word)

	embeddings = list()
	for i in range(0, 2393):
		embeddings.append(embeddings_temp[i])

	return train, test, word_to_id, id_to_word, np.asarray(embeddings)