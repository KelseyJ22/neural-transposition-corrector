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
max_word_len = 10
char_embedding_size = 300
	

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


def build_embeddings(char_embeddings, train):
	embedding_lookup = dict()
	curr_id = 0
	word_embeddings = list()

	for i in range(0, len(train)):
		inp = train[i][0]
		label = train[i][1]
		for word in inp:
			if word not in embedding_lookup:
				embedding_lookup[word] = curr_id
				curr_id += 1
				word_embedding = list()
				for i in range(0, max_word_len):
					if i < len(word):
						char = word[i]
						if char in char_embeddings:
							embed = char_embeddings[char]
						else:
							embed = np.zeros((char_embedding_size))
						word_embedding.append(embed)
					else:
						word_embedding.append(np.zeros((char_embedding_size)))
				word_embeddings.append(np.asarray(word_embedding))

		for word in label:
			if word not in embedding_lookup:
				embedding_lookup[word] = curr_id
				curr_id += 1
				word_embedding = list()
				for i in range(0, max_word_len):
					if i < len(word):
						char = word[i]
						if char in char_embeddings:
							embed = char_embeddings[char]
						else:
							embed = np.zeros((char_embedding_size))
						word_embedding.append(embed)
					else:
						word_embedding.append(np.zeros((char_embedding_size)))

				word_embeddings.append(np.asarray(word_embedding))

	return np.asarray(word_embeddings), embedding_lookup


def load_from_file():
	test_file = '../Data/test_local'
	test_file_obj = open(test_file, 'r')
	test = pickle.load(test_file_obj)
	test_file_obj.close()

	train_file = '../Data/train_local'
	train_file_obj = open(train_file, 'r')
	train = pickle.load(train_file_obj)
	train_file_obj.close()

	word2id_file = '../Data/word2id'
	word2id_file_obj = open(word2id_file, 'r')
	word_to_id = pickle.load(word2id_file_obj)
	word2id_file_obj.close()

	id_to_word = dict()
	for word in word_to_id:
		id_to_word[word_to_id[word]] = word
	
	embeddings_obj = open('../Data/char_embeddings.txt', 'r')
	char_embeddings = dict()
	for line in embeddings_obj:
		split = line.split(' ')
		key = split[0].strip()
		value = split[1:]
		res = list()
		for entry in value:
			res.append(float(entry))
		char_embeddings[key] = np.asarray(res)

	embeddings, embedding_lookup = build_embeddings(char_embeddings, train)

	return train, test, word_to_id, id_to_word, embedding_lookup, embeddings