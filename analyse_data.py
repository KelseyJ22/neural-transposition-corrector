import re


def clean(word):
	word = word.strip().lower()
	word = word.replace(',', '')
	word = word.replace(';', '')
	word = word.replace(':', '')
	word = word.replace(')', '')
	word = word.replace('(', '')
	word = word.replace(']', '')
	word = word.replace('[', '')
	word = word.replace('-', '')
	word = word.replace('<u>', '')
	word = word.replace("\"", '')
	return str(word)


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

		parsed.append(sent)
	return parsed

o = open('Data/movie_lines.txt', 'r')
lines = o.readlines()
o.close()

print 'total lines', len(lines)

dataset = list()
frequencies = dict()
sent_dist = dict()
word_dist = dict()
total_words = 0
for line in lines:
	sentences = parse(line)
	for sentence in sentences:
			dataset.append(sentence)
			
			if len(sentence) in sent_dist:
				sent_dist[len(sentence)] += 1
			else:
				sent_dist[len(sentence)] = 1

			for word in sentence:
				w = clean(word)
				if len(w) > 0:
					total_words += 1
					if w in frequencies:
						frequencies[w] += 1
					else:
						frequencies[w] = 1

					if len(w) in word_dist:
						word_dist[len(w)] += 1
					else:
						word_dist[len(w)] = 1

print 'read in ', len(dataset), 'samples'
print 'found a total of', len(frequencies), 'words'

total_len = 0
max_len = 0
for sentence in dataset:
	total_len += len(sentence)
	if len(sentence) > max_len:
		max_len = len(sentence)


total_word_len = 0
max_word_len = 0
for word in frequencies:
	total_word_len += len(word)
	if len(word) > max_word_len:
		max_word_len = len(word)

print 'average sentence length is', total_len/float(len(dataset)), 'with maximum', max_len
print 'average word length is', total_word_len/float(len(frequencies)), 'with maximum', max_word_len

print 'sentence length frequencies:'
cumulative = 0
for k in sorted(sent_dist):
	cumulative += sent_dist[k]
	print k, sent_dist[k], 'percent up to here:', cumulative/float(len(dataset))

print 'word length frequencies:'
cumulative = 0
for k in sorted(word_dist):
	cumulative += word_dist[k]
	print k, word_dist[k], 'percent up to here:', cumulative/float(total_words)