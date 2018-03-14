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


def load_test(fname):
	test_file = 'data/train_weighted_' + fname
	test_file_obj = open(test_file, 'rb')
	test = pickle.load(test_file_obj)
	test_file_obj.close()
	print 'loaded test data from', fname
	return test


def load_from_file(fname):
	test_file = 'data/test_weighted_' + fname
	test_file_obj = open(test_file, 'rb')
	test = pickle.load(test_file_obj)
	test_file_obj.close()
	print 'loaded test data'

	train_file = 'data/train_weighted_' + fname
	train_file_obj = open(train_file, 'rb')
	train = pickle.load(train_file_obj)
	train_file_obj.close()
	print 'loaded train data'


	id2word_file = 'data/id2word_weighted_' + fname
	id2word_file_obj = open(id2word_file, 'rb')
	id_to_word = pickle.load(id2word_file_obj)
	id2word_file_obj.close()
	print 'loaded id_to_word lookup'

	embedding_lookup_file = 'data/embedding_lookup_weighted_' + fname
	embedding_lookup_file_obj = open(embedding_lookup_file, 'rb')
	embedding_lookup = pickle.load(embedding_lookup_file_obj)
	embedding_lookup_file_obj.close()
	print 'loaded embedding lookup counts'


	embed_file = 'data/embeddings_weighted_' + fname
	embed_file_obj = open(embed_file, 'rb')
	embeddings = pickle.load(embed_file_obj)
	embed_file_obj.close()
	print 'loaded embeddings'

	return train, test, id_to_word, embedding_lookup, embeddings


def load_from_file_basic():
	test_file = '../Data/test_all'
	test_file_obj = open(test_file, 'rb')
	test = pickle.load(test_file_obj)
	test_file_obj.close()
	print 'loaded test data'

	train_file = '../Data/train_all'
	train_file_obj = open(train_file, 'rb')
	train = pickle.load(train_file_obj)
	train_file_obj.close()
	print 'loaded train data'

	word2id_file = '../Data/word2id_all'
	word2id_file_obj = open(word2id_file, 'rb')
	word_to_id = pickle.load(word2id_file_obj)
	word2id_file_obj.close()
	print 'loaded word_to_id lookup'

	id2word_file = '../Data/id2word_all'
	id2word_file_obj = open(id2word_file, 'rb')
	id_to_word = pickle.load(id2word_file_obj)
	id2word_file_obj.close()
	print 'loaded id_to_word lookup'

	embeddings_temp = dict()
	id_to_word = dict()
	for word in word_to_id:
		id_to_word[word_to_id[word]] = word
		embeddings_temp[word_to_id[word]] = create_embedding(word)

	embeddings = list()
	for i in range(0, 2393):
		embeddings.append(embeddings_temp[i])

	return train, test, word_to_id, id_to_word, np.asarray(embeddings)