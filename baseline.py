
import nltk
import gensim
import pandas as pd
import numpy as np
import sys
import codecs
import math

from sklearn.linear_model import LogisticRegression


def average_vecs(w2v, filename):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	with open(filename) as f:
		df = pd.read_csv(f, dtype={'text': str})
		samples = []
		labels = []
		reviews = df['text']
		num_rows = len(df)
		for i, review in enumerate(reviews):
			if i % 1000 == 0:
				print str(i) + '/' + str(num_rows)
			if type(review) is not str:
				continue
			labels.append(df.iloc[i]['style'])
			summed = np.zeros((100))
			review_tokenized = nltk.word_tokenize(review.lower())
			for token in review_tokenized:
				if token in w2v.wv.vocab:
					summed += w2v[token]
			summed /= num_rows
			samples.append(summed)
		return np.array(samples), labels


def main():
	w2vFile = sys.argv[1]
	w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)
	train_x, train_y = average_vecs(w2v, 'beers_train.csv')
	dev_x, dev_y = average_vecs(w2v, 'beers_dev.csv')
	
	baseline = LogisticRegression()
	baseline.fit(train_x, train_y)
	y_hat = baseline.predict(dev_x)

	correct = 0.0
	total = len(y_hat)

	for i in range(len(y_hat)):
		if dev_y[i] == y_hat[i]:
			correct += 1
	acc = correct / total
	print 'accuracy: ' + str(acc)



main()