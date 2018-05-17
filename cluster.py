
import nltk
import gensim
import pandas as pd
import numpy as np
import sys

from sklearn.cluster import KMeans



def average_vecs(w2v):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	with open('beers.csv') as f:
		df = pd.read_csv(f)
		styles = df['style'].unique()[:-1]
		all_styles = np.zeros((89, 100))
		for i, s in enumerate(styles):
			all_beers = df.loc[df['style'] == s]
			summed = np.zeros((100))
			top_thousand = all_beers.head(1000)
			num_rows = len(top_thousand)
			for index, row in top_thousand.iterrows():
				review = row['text']
				review_tokenized = nltk.word_tokenize(review.lower())
				for token in review_tokenized:
					if token in w2v.wv.vocab:
						summed += w2v[token]
			summed /= num_rows
			all_styles[i] = summed
		return all_styles, styles

def cluster(all_styles):
	kmeans = KMeans()
	kmeans.fit(all_styles)
	return kmeans.labels_


def main():
	w2vFile = sys.argv[1]
	w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)
	all_styles, styles = average_vecs(w2v)
	labels = cluster(all_styles)
	clusters = {}
	for i in range(8):
		clusters[i] = []
	for j in range(len(labels)):
			clusters[labels[j]].append(styles[j])
	print(clusters)



main()