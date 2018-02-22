import utils

train, test, frequencies = utils.load_data('Data/movie_lines.txt')

keep_20 = list()
keep_40 = list()
keep_60 = list()
keep_80 = list()

for word in frequencies:
	if frequencies[word] > 20:
		keep_20.append((word, frequencies[word]))
	if frequencies[word] > 40:
		keep_40.append((word, frequencies[word]))
	if frequencies[word] > 60:
		keep_60.append((word, frequencies[word]))
	if frequencies[word] > 80:
		keep_80.append((word, frequencies[word]))

print 'words more frequent than 20', len(keep_20)
print 'words more frequent than 40', len(keep_40)
print 'words more frequent than 60', len(keep_60)
print 'words more frequent than 80', len(keep_80)

for entry in keep_80:
	print entry