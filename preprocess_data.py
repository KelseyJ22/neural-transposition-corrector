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


def load_word_lookup_all(train, test, frequencies):
	word_to_id2 = dict()
	id_to_word2 = dict()
	embeddings = list()
	vocab_count = 0
	embedding_count = 0
	for example in train:
		sentence = example[0]
		label = example[1]
		for i in range(0, len(sentence)):
			label_word = label[i]
			scrambled_word = sentence[i]
			if label_word in frequencies and frequencies[label_word] > 80:
				if len(label_word) > 0:
					if label_word not in word_to_id2:
						word_to_id2[label_word] = embedding_count
						id_to_word2[embedding_count] = label_word
						vocab_count += 1
						embedding_count += 1
						embeddings.append(utils.create_weighted_embedding(label_word))

					if scrambled_word not in word_to_id2:
						word_to_id2[scrambled_word] = embedding_count
						id_to_word2[embedding_count] = scrambled_word
						embedding_count += 1
						embeddings.append(utils.create_weighted_embedding(scrambled_word))

	for example in train:
		sentence = example[0]
		label = example[1]
		for i in range(0, len(sentence)):
			label_word = label[i]
			scrambled_word = sentence[i]
			if label_word in frequencies and frequencies[label_word] > 80:
				if len(label_word) > 0:
					if label_word not in word_to_id2:
						word_to_id2[label_word] = embedding_count
						id_to_word2[embedding_count] = label_word
						vocab_count += 1
						embedding_count += 1
						embeddings.append(utils.create_weighted_embedding(label_word))
						
					if scrambled_word not in word_to_id2:
						word_to_id2[scrambled_word] = embedding_count
						id_to_word2[embedding_count] = scrambled_word
						embedding_count += 1
						embeddings.append(utils.create_weighted_embedding(scrambled_word))


	word_to_id2['<UNK>'] = embedding_count
	word_to_id2['0'] = embedding_count + 1
	id_to_word2[embedding_count] = '<UNK>'
	id_to_word2[embedding_count + 1] = '0'
	embeddings.append(np.zeros((len(embeddings[-1])))) # to correspond to UNK
	embeddings.append(np.zeros((len(embeddings[-1])))) # to correspond to 0
	return word_to_id2, id_to_word2, embeddings
	

train, test, frequencies = utils.load_data('Data/movie_lines.txt')

embedding_lookup, reverse_embedding_lookup, embeddings = load_word_lookup_all(train, test, frequencies)

word_to_id, id_to_word, _ = load_word_lookup(frequencies)

test_file = 'Data/test_all'
test_file_obj = open(test_file, 'wb')
pickle.dump(test, test_file_obj)
test_file_obj.close()

train_file = 'Data/train_all'
train_file_obj = open(train_file, 'wb')
pickle.dump(train, train_file_obj)
train_file_obj.close()

word2id_file = 'Data/word2id_all'
word2id_file_obj = open(word2id_file, 'wb')
pickle.dump(word_to_id, word2id_file_obj)
word2id_file_obj.close()

id2word_file = 'Data/id2word_all'
id2word_file_obj = open(id2word_file, 'wb')
pickle.dump(word_to_id, id2word_file_obj)
id2word_file_obj.close()

embedding_lookup_file = 'Data/embedding_lookup_all'
embedding_lookup_file_obj = open(embedding_lookup_file, 'wb')
pickle.dump(word_to_id, embedding_lookup_file_obj)
embedding_lookup_file_obj.close()

reverse_embedding_lookup_file = 'Data/reverse_embedding_lookup_all'
reverse_embedding_lookup_file_obj = open(reverse_embedding_lookup_file, 'wb')
pickle.dump(word_to_id, reverse_embedding_lookup_file_obj)
reverse_embedding_lookup_file_obj.close()

embed_file = 'Data/embeddings_all'
embed_file_obj = open(embed_file, 'wb')
pickle.dump(embeddings, embed_file_obj)
embed_file_obj.close()