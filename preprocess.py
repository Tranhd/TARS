"""
Class that loads and preprocesses the Jokes.
"""
import pandas as pd
import re
import nltk
import numpy as np
import json

class Preprocess(object):
	"""docstring for Preprocess.
	
	Attributes:
	    data (array): The preprocessed dataset.
	    index2word (array): Mapping from the indices to the words.
	    max_len (int): maximum length of jokes used as data.
	    PAD (str): Pad token.
	    sentence_end_token (str): End of sentence token.
	    sentence_start_token (str): Start of sentence token.
	    unknown_token (str): Unknown token.
	    unknowns (list): Description
	    vocab_size (int): The vocabulary size.
	    word2index (dictionary): Mapping from the words to the indices.
	"""
	def __init__(self, max_len = 12):
		"""
		Loads the Jokes and preprocess.
		
		Args:
		    max_len (int, optional): maximum length of jokes used as data.
		"""
		# Definitions.

		self.unknown_token = "UNK"
		self.sentence_start_token = "GO"
		self.sentence_end_token = "END"
		self.PAD = "PAD"
		self.max_len = max_len
		self.unknowns = []
		
		try:
			dataframe = pd.read_csv('shortjokes.csv')
			print("Jokes loaded successfully")
		except Exception as e:
			print("Could not load Jokes")
			raise

		filtered_jokes = self.get_short_jokes(dataframe)
		tokenized_jokes = self.tokenize_words(filtered_jokes)
		self.create_vocab(tokenized_jokes)
		self.data = self.encode(tokenized_jokes)

	def get_short_jokes(self, dataframe):
		"""
		Remove jokes with length over max_len characters.
		
		Returns:
		    dataframe: The filtered jokes.
		
		Args:
		    dataframe (dataframe): The non-filtered jokes.
		"""
		f = lambda x: (len(x.split())) 
		filtered_jokes = dataframe[dataframe['Joke'].apply(f) < self.max_len]
		print("Jokes above a length of %d removed" %self.max_len)
		return filtered_jokes

	def tokenize_words(self, jokes):
		"""Adds the sentance end and sentance start tokens split the jokes into arrays of words.
		
		Args:
		    jokes (dataframe): The jokes to be word tokenized.
		
		Returns:
		    array: The tokenized words.
		"""
		jokes = jokes['Joke'].values
		jokes_tok = ["%s %s %s" % (self.sentence_start_token, x.lower(), self.sentence_end_token) for x in jokes]
		tokenized_jokes = [nltk.word_tokenize(sentence) for sentence in jokes_tok]
		print("Sentence end and start tokens added and jokes word-tokenized")
		return tokenized_jokes

	def create_vocab(self, tokenized_jokes):
		"""Calculates less frequent words and creates the vocabulary and mappings.
		
		Args:
		    tokenized_jokes (array): The current jokes.
		"""
		fdist = nltk.FreqDist()
		for sent in tokenized_jokes:
		    for word in sent:
		        fdist[word] += 1
		print("Found %d unique word tokens." % len(fdist.items()))

		total_count = 0
		for count in fdist.values():
		    total_count +=count
		print("Found %d total word tokens." %total_count)

		vocab = fdist.most_common()
		current_count = 0
		for vocab_size, count in enumerate(vocab):
		    current_count += count[1]
		    if current_count/total_count > 0.98:
		        print("Use %d most common words to account for the 98 percent of the total word counts" % (vocab_size))
		        break
		vocab = fdist.most_common(vocab_size)
		self.vocab_size = vocab_size + 2
		self.index2word = [x[0] for x in vocab]
		self.index2word.insert(0, self.PAD)
		self.index2word.append(self.unknown_token)
		self.word2index = dict([(w,i) for i,w in enumerate(self.index2word)])
		print("The least frequent word in our vocabulary is '%s' and appeared %d time(s)." % (vocab[-1][0], vocab[-1][1]))
		print("The vocabulary is of size %d" % self.vocab_size)

	def encode(self, tokenized_jokes):
		"""Adds the unknown token and creates the final dataset.
		
		Args:
		    tokenized_jokes (array): The tokenized jokes.
		
		Returns:
		    array: The final encoded data set.
		"""
		for i, sent in enumerate(tokenized_jokes):
		    tokenized_jokes[i] = [w if w in self.word2index else self.unknown_token for w in sent]
		data = np.asarray([[self.word2index[w] for w in joke] for joke in tokenized_jokes])
		print("\nJoke after Pre-processing: '%s'" % tokenized_jokes[5])
		print("\nJoke after encoding:", data[5])
		return data

	def get_data(self):
		return self.data

	def i2w(self, index):
		return self.index2word[index]

	def w2i(self, word):
		return self.word2index[word]

	def dump(self):
		json.dump(self.index2word, open("i2w.txt",'w'))
		json.dump(self.word2index, open("w2i.txt",'w'))
		json.dump(self.vocab_size, open("vocab_size.txt",'w'))
