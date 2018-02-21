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
		line = line[start:]
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
	res = np.zeros([len(input_list)])
	for i in range(0, len(input_list)):
		res[i] = float(input_list[i])
	return res


def load_embeddings(data):
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
	new_word = word[0], shuffle_string(inner_chars), word[-1]

	return new_word

def true_len(word):
	count = 0
	real_chars = set() # TODO
	for char in word:
		if char in real_chars:
			count += 1
	return count


def generate_errors(data):
	errored = list()
	for sentence in data:
		new_sentence = list()
		for word in sentence:
			if true_len(word) > 3:
				gen_error = random.randint(0, 3)
				if gen_error == 0:
					word = add_error(word)
			new_sentence.append(word)
		errored.append((new_sentence, sentence)) # input, output pair
	return errored


def load_data(fname):
	# read in data
	data = open(fname, 'r')
	dataset = list()
	for line in data.readlines():
		text_data = parse(line)
		for sentence in text_data:
			dataset.append(sentence)
	print 'read in ', len(dataset), 'samples'

	# add transposition errors
	parsed_data = generate_errors(dataset)

	# train/test split
	train, test = parsed_data[0:10000], parsed_data[10000:] # TODO: define this split more intelligently
	
	return train, test


"""def reshape(batch):
	# TODO: there has got to be a better way to do this
	transposed_encoder_seqs = []
    for i in range(0, max_length):
        transposed_encoder_seq = []
        for j in range(0, batch_size):
            transposed_encoder_seq.append(batch.encoder_seq[j][i])
        transposed_encoder_seqs.append(transposed_encoder_seq)
    batch.encoder_seq = transposed_encoder_seqs

    transposed_decoder_seqs = []
    transposed_target_seq = []
    transposed_weight = []
    for i in range(0, max_length):
        transposed_decoder_seq = []
        transposed_target_seq = []
        weightT = []
        for j in range(batchSize):
            transposed_decoder_seq.append(batch.decoder_seq[j][i])
            transposed_target_seq.append(batch.target_seq[j][i])
            transposed_weight.append(batch.weights[j][i])
        transposed_decoder_seqs.append(transposed_decoder_seq)
        transposed_target_seqs.append(transposed_target_seq)
        transposed_weights.append(transposed_weight)
    batch.decoder_seq = transposed_decoder_seqs
    batch.target_seq = transposed_target_seqs
    batch.weights = transposed_weights
    return batch"""


def batchify(data, batch_size):
	batch = Batch()
	for i in range(0, batch_size):
		sample = data[i]
		batch.encoder_seq.append(list(reversed(sample[0])))
		batch.decoder_seq.append([start + sample[1] + end])
		batch.target_seq.append(batch.decoder_seq[-1][1:])

		batch.encoder_seq[i] = [pad] * (max_length - len(batch.encoder_seq[i])) + batch.encoder_seq[i] # left padding
		batch.weights.append([1.0] * len(batch.target_seq[i]) + [0.0] * (max_length - len(batch.target_seq[i]))) # right padding
		batch.decoder_seq[i] = batch.decoder_seq[i] + [pad] * (max_length - len(batch.decoder_seq[i])) # right padding
		batch.target_seq[i] = batch.target_seq[i] + [pad] * (max_length - len(batch.target_seq[i])) # right pad

	return reshape(batch)



def get_batch(data, sample_size, batch_size):
	for i in range(0, sample_size, batch_size):
		yield data[i:min(i + batch_size, sample_size)] 


def get_batches(data, sample_size, batch_size):
	random.shuffle(data) # reshuffle for each iteration
	for batch in get_batch(data, sample_size, batch_size):
		batches.append(batchify(batch, batch_size))

	return batches