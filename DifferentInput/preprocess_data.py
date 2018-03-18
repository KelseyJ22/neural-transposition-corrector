import numpy as np 
import helper
import pickle


def gen_word_lookup_simple(frequencies):
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
					embeddings.append(helper.create_embedding(word))

	word_to_id['<UNK>'] = current_count
	word_to_id['0'] = current_count + 1
	id_to_word[current_count] = '<UNK>'
	id_to_word[current_count + 1] = '0'
	embeddings.append(np.zeros((78))) # to correspond to UNK
	embeddings.append(np.zeros((78))) # to correspond to 0
	return word_to_id, id_to_word, embeddings


def gen_word_lookup_weighted(train, test, frequencies):
	embedding_lookup = dict()
	reverse_embedding_lookup = dict()
	embeddings = list()
	vocab_count = 0
	embedding_count = 0
	for file in [test, train]:
		for example in train:
			sentence = example[0]
			label = example[1]
			for i in range(0, len(sentence)):
				label_word = label[i]
				scrambled_word = sentence[i]
				if label_word in frequencies and frequencies[label_word] > 80:
					if len(label_word) > 0:
						if label_word not in embedding_lookup:
							embedding_lookup[label_word] = embedding_count
							reverse_embedding_lookup[embedding_count] = label_word
							vocab_count += 1
							embedding_count += 1
							embeddings.append(helper.create_weighted_embedding(label_word))

						if scrambled_word not in embedding_lookup:
							embedding_lookup[scrambled_word] = embedding_count
							reverse_embedding_lookup[embedding_count] = scrambled_word
							embedding_count += 1
							embeddings.append(helper.create_weighted_embedding(scrambled_word))


	embedding_lookup['<UNK>'] = embedding_count
	embedding_lookup['0'] = embedding_count + 1
	reverse_embedding_lookup[embedding_count] = '<UNK>'
	reverse_embedding_lookup[embedding_count + 1] = '0'
	embeddings.append(np.zeros((len(embeddings[-1])))) # to correspond to UNK
	embeddings.append(np.zeros((len(embeddings[-1])))) # to correspond to 0
	return embedding_lookup, reverse_embedding_lookup, embeddings
	

train, test, frequencies = helper.load_data('data/movie_lines.txt')

embedding_lookup, reverse_embedding_lookup, embeddings = gen_word_lookup_weighted(train, test, frequencies)

word_to_id, id_to_word, _ = gen_word_lookup_simple(frequencies)

fname = 'shuffle'

test_file = 'data/test_weighted_no_context_' + fname
test_file_obj = open(test_file, 'wb')
pickle.dump(test, test_file_obj)
test_file_obj.close()

train_file = 'data/train_weighted_no_context_' + fname
train_file_obj = open(train_file, 'wb')
pickle.dump(train, train_file_obj)
train_file_obj.close()

word2id_file = 'data/word2id_weighted_no_context_' + fname
word2id_file_obj = open(word2id_file, 'wb')
pickle.dump(word_to_id, word2id_file_obj)
word2id_file_obj.close()

id2word_file = 'data/id2word_weighted_no_context_' + fname
id2word_file_obj = open(id2word_file, 'wb')
pickle.dump(id_to_word, id2word_file_obj)
id2word_file_obj.close()

embedding_lookup_file = 'data/embedding_lookup_weighted_no_context_' + fname
embedding_lookup_file_obj = open(embedding_lookup_file, 'wb')
pickle.dump(embedding_lookup, embedding_lookup_file_obj)
embedding_lookup_file_obj.close()

reverse_embedding_lookup_file = 'data/reverse_embedding_lookup_weighted_no_context_' + fname
reverse_embedding_lookup_file_obj = open(reverse_embedding_lookup_file, 'wb')
pickle.dump(reverse_embedding_lookup, reverse_embedding_lookup_file_obj)
reverse_embedding_lookup_file_obj.close()

embed_file = 'data/embeddings_weighted_no_context_' + fname
embed_file_obj = open(embed_file, 'wb')
pickle.dump(embeddings, embed_file_obj)
embed_file_obj.close()