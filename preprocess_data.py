import numpy as np 
import utils

def load_word_lookup(frequencies):
	lookups = dict()
	current_count = 1
	for word in frequencies:
		if frequencies[word] > 80:
			if len(word) > 0:
				if word not in lookups:
					lookups[word] = current_count
					current_count += 1

	# map all non-word characters to 0
	lookups['<s>'] = 0
	lookups['</s>'] = 0
	lookups['<UNK>'] = 0
	lookups['0'] = 0
	return lookups
	

train, test, frequencies = utils.load_data('Data/movie_lines.txt')

word_to_id = load_word_lookup(frequencies)


f = open('train.txt','w')
for i in range(0, len(train)):
	sentence = ''
	for word in train[i][0]:
		sentence += word + ','
	sentence = sentence[:-1]
	sentence += '|'
	for word in train[i][1]:
		sentence += word + ','
	sentence = sentence[:-1]
	sentence += '\n'
	f.write(sentence)
f.close()

f = open('test.txt','w')
for i in range(0, len(test)):
	sentence = ''
	for word in test[i][0]:
		sentence += word + ','
	sentence = sentence[:-1]
	sentence += '|'
	for word in test[i][1]:
		sentence += word + ','
	sentence = sentence[:-1]
	sentence += '\n'
	f.write(sentence)
f.close()

f = open('word2id.txt', 'w')
for word in word_to_id:
	f.write(word + ':' + str(word_to_id[word]) + '\n')
f.close()