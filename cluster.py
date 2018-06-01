
import nltk
import gensim
import pandas as pd
import numpy as np
import sys
import pylab as pl
import codecs

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



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

def add_cluster_csv(cluster):
	train = pd.read_csv('beers_train.csv')
	dev = pd.read_csv('beers_dev.csv')
	test = pd.read_csv('beers_test.csv')

	train_clusters = []
	dev_clusters = []
	test_clusters = []

	for index, row in train.iterrows():
		s = row['style']
		if type(s) is not str:
			train_clusters.append(float('nan'))
			continue
		string_nobackslashes = codecs.decode(s, 'unicode_escape')
		string_nobackslashes = string_nobackslashes.encode('ISO-8859-1')
		correct_string = string_nobackslashes.replace(b'\xc3\xa9', 'e').replace(b'\xc3\xb6', 'o').replace(b'&#40;IPA&#41;', '').replace(b'&#40;Witbier&#41;', '').replace('\xc3\xa4', 'a').replace(b'\xc3\xa8', 'e').replace('b\'', '').replace('\'', '').strip()
		correct_string = correct_string.decode('ascii')
		train_clusters.append(cluster[correct_string] + 1)

	train['cluster'] = train_clusters

	for index, row in dev.iterrows():
		s = row['style']
		if type(s) is not str:
			dev_clusters.append(float('nan'))
			continue
		string_nobackslashes = codecs.decode(s, 'unicode_escape')
		string_nobackslashes = string_nobackslashes.encode('ISO-8859-1')
		correct_string = string_nobackslashes.replace(b'\xc3\xa9', 'e').replace(b'\xc3\xb6', 'o').replace(b'&#40;IPA&#41;', '').replace(b'&#40;Witbier&#41;', '').replace('\xc3\xa4', 'a').replace(b'\xc3\xa8', 'e').replace('b\'', '').replace('\'', '').strip()
		correct_string = correct_string.decode('ascii')
		dev_clusters.append(cluster[correct_string] + 1)

	dev['cluster'] = dev_clusters

	for index, row in test.iterrows():
		s = row['style']
		if type(s) is not str:
			test_clusters.append(float('nan'))
			continue
		string_nobackslashes = codecs.decode(s, 'unicode_escape')
		string_nobackslashes = string_nobackslashes.encode('ISO-8859-1')
		correct_string = string_nobackslashes.replace(b'\xc3\xa9', 'e').replace(b'\xc3\xb6', 'o').replace(b'&#40;IPA&#41;', '').replace(b'&#40;Witbier&#41;', '').replace('\xc3\xa4', 'a').replace(b'\xc3\xa8', 'e').replace('b\'', '').replace('\'', '').strip()
		correct_string = correct_string.decode('ascii')
		test_clusters.append(cluster[correct_string] + 1)

	test['cluster'] = test_clusters

	train.to_csv('beers_train_cluster.csv')
	dev.to_csv('beers_dev_cluster.csv')
	test.to_csv('beers_test_cluster.csv')


def main():
	replacements = {'\xc3\xa9': 'e', '\xc3\xb6': 'o', '&#40;IPA&#41;': ''}
	w2vFile = sys.argv[1]
	w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)
	all_styles, styles = average_vecs(w2v)
	new_styles = []
	for s in styles:
		string_nobackslashes = codecs.decode(s, 'unicode_escape')
		string_nobackslashes = string_nobackslashes.encode('ISO-8859-1')
		correct_string = string_nobackslashes.replace(b'\xc3\xa9', 'e').replace(b'\xc3\xb6', 'o').replace(b'&#40;IPA&#41;', '').replace(b'&#40;Witbier&#41;', '').replace('\xc3\xa4', 'a').replace(b'\xc3\xa8', 'e').replace('b\'', '').replace('\'', '').strip()
		correct_string = correct_string.decode('ascii')
		new_styles.append(correct_string)
	print(new_styles)
	styles = new_styles
	labels = cluster(all_styles)
	cluster_key = {}
	clusters = {}
	for i in range(8):
		clusters[i] = []
	for j in range(len(labels)):
		cluster_key[styles[j]] = labels[j]
		clusters[labels[j]].append(styles[j])
	print(clusters)

	add_cluster_csv(cluster_key)

	pca = PCA(n_components=2).fit(all_styles)
	pca_2d = pca.transform(all_styles)

	for i in range(pca_2d.shape[0]):
		if labels[i] == 0:
			c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r')
		elif labels[i] == 1:
			c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g')
		elif labels[i] == 2:
			c3 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='b')
		elif labels[i] == 3:
			c4 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='c')
		elif labels[i] == 4:
			c5 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='m')
		elif labels[i] == 5:
			c6 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='y')
		elif labels[i] == 6:
			c7 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='k')
		elif labels[i] == 7:
			c8 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='orange')
		pl.annotate(styles[i], xy=(pca_2d[i,0], pca_2d[i,1]))
	pl.legend([c1, c2, c3, c4, c5, c6, c7, c8], ['1', '2', '3', '4', '5', '6', '7', '8'])
	pl.show()



main()