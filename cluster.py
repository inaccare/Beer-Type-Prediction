
import nltk
import gensim
import pandas as pd
import numpy as np
import sys



def average_vecs(w2v):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	with open('beers.csv') as f:
		df = pd.read_csv(f)
		styles = df['style'].unique()
		all_styles = np.zeros((89, 100))
		for i, s in enumerate(styles):
			all_beers = df.loc[df['style'] == s]
			summed = np.zeros((100))
			top_thousand = all_beers.head(1000)
			num_rows = len(top_thousand)
			for index, row in top_thousand.iterrows():
				review = row['text'].encode('utf-8').decode('utf-8')
				review_tokenized = nltk.word_tokenize(review)
				for token in review_tokenized:
					print(token)
					summed += w2v[token]
			summed /= num_rows
			print(summed)
			all_styles[i] = summed
		print(all_styles)

def main():
	w2vFile = sys.argv[1]
	w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)
	average_vecs(w2v)



main()