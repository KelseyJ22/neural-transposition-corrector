import numpy as np 
import utils
import pickle


def load_word_lookup(frequencies):
	word_to_id = dict()
	id_to_word = dict()
	embeddings = list()
	current_count = 0
	for word in frequencies:
		if frequencies[word] > 80:
			if len(word) > 0:
				if word not in word_to_id:
					word_to_id[word] = current_count
					id_to_word[current_count] = word
					current_count += 1
					embeddings.append(utils.create_embedding(word))

	word_to_id['<UNK>'] = current_count
	word_to_id['0'] = current_count + 1
	id_to_word[current_count] = '<UNK>'
	id_to_word[current_count + 1] = '0'
	embeddings.append(np.zeros((78))) # to correspond to UNK
	embeddings.append(np.zeros((78))) # to correspond to 0
	return word_to_id, id_to_word, embeddings
	

train, test, frequencies = utils.load_data('Data/movie_lines.txt')

word_to_id, id_to_word, embeddings = load_word_lookup(frequencies)

test_file = 'Data/test'
test_file_obj = open(test_file, 'wb')
pickle.dump(test, test_file_obj)
test_file_obj.close()

train_file = 'Data/train'
train_file_obj = open(train_file, 'wb')
pickle.dump(train, train_file_obj)
train_file_obj.close()

word2id_file = 'Data/word2id'
word2id_file_obj = open(word2id_file, 'wb')
pickle.dump(word_to_id, word2id_file_obj)
word2id_file_obj.close()

id2word_file = 'Data/id2word'
id2word_file_obj = open(id2word_file, 'wb')
pickle.dump(word_to_id, id2word_file_obj)
id2word_file_obj.close()

embed_file = 'Data/embeddings'
embed_file_obj = open(embed_file, 'wb')
pickle.dump(word_to_id, embed_file_obj)
embed_file_obj.close()