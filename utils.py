import numpy as np
import nltk
import math
import random

# TODO: define all of these
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

def load_embeddings(data):
	# TODO complete this
	pass


def generate_errors(data):
	errored = list()
	for sentence in data:
		new_sentence = list()
		for word in sentence:
			if len(word) > 3:
				gen_error = random.randint(0, 3)
				if gen_error == 0:
					word = add_error(word)
			new_sentence.append(word)
		errored.append((new_sentence, sentence)) # input, output pair
	return errored


def load_data():
	# read in data
	# TODO choose dataset and read in
	# add transposition errors
	parsed_data = generate_errors(data)

	# train/test split
	train, test = data[0:10000], data[10000:] # TODO: define this split more intelligently
	
	return train, test


def reshape(batch):
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
    return batch


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

	return batch