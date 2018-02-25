import numpy as np 
import utils
import pickle


def load_word_lookup(frequencies):
	lookups = dict()
	current_count = 0
	for word in frequencies:
		if frequencies[word] > 80:
			if len(word) > 0:
				if word not in lookups:
					lookups[word] = current_count
					current_count += 1

	lookups['<UNK>'] = 2000
	lookups['0'] = 2001
	return lookups
	

train, test, frequencies = utils.load_data('Data/movie_lines.txt')

word_to_id = load_word_lookup(frequencies)

test_file = 'test'
test_file_obj = open(test_file, 'wb')
pickle.dump(test, test_file_obj)
test_file_obj.close()

train_file = 'train'
train_file_obj = open(train_file, 'wb')
pickle.dump(train, train_file_obj)
train_file_obj.close()

word2id_file = 'word2id'
word2id_file_obj = open(word2id_file, 'wb')
pickle.dump(word_to_id, word2id_file_obj)
word2id_file_obj.close()