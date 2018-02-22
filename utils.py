import numpy as np
import math
import random
import re

start = '<s>'
end = '</s>'
pad = '0'
max_length = 10
unknown = '<UNK>'


class Batch:
	def __init__(self):
		self.encoder_seq = []
		self.decoder_seq = []
		self.target_seq = []
		self.weights = []


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
				if frequencies[word] > 80:
					cleaned_sentence.append(word)
				else:
					cleaned_sentence.append(UNK)
			if is_transposable(word):
				gen_error = random.randint(0, 3)
				if gen_error == 0:
					word = add_error(word)
			word = clean(word)
			if len(word) > 0:
				if frequencies[word] > 80:
					new_sentence.append(word)
				else:
					new_sentence.append(UNK)

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
	batch = Batch()
	for i in range(0, len(data)):
		sample = data[i]
		batch.encoder_seq.append(list(reversed(sample[0])))
		batch.decoder_seq.append([start] + sample[1] + [end])
		batch.target_seq.append(batch.decoder_seq[-1][1:]) # shift over by one

		batch.encoder_seq[i] = [pad] * (max_length - len(batch.encoder_seq[i])) + batch.encoder_seq[i] # left padding
		batch.weights.append([1.0] * len(batch.target_seq[i]) + [0.0] * (max_length - len(batch.target_seq[i]))) # right padding
		batch.decoder_seq[i] = batch.decoder_seq[i] + [pad] * (max_length - len(batch.decoder_seq[i])) # right padding
		batch.target_seq[i] = batch.target_seq[i] + [pad] * (max_length - len(batch.target_seq[i])) # right pad
	return batch


def get_batch(data, batch_size):
	for i in range(0, batch_size):
		yield data[i:i + batch_size] 


def get_batches(data, batch_size):
	batches = list()
	random.shuffle(data) # reshuffle for each iteration
	for batch in get_batch(data, batch_size):
		batches.append(batchify(batch, batch_size))

	return batches