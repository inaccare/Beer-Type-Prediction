import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd


bins = 30
beer_file = 'beers.csv'

x = []

with open(beer_file) as f:
	df = pd.read_csv(f)
	length = len(df['text'])
	for i, row in enumerate(df['text'][:-1]):
		if i % 100000:
			print str(i) + '/' + str(length)
		x.append(len(row))

	vals = np.asarray(x, dtype=np.int32)

	plt.xlabel('Tokens')
	plt.ylabel('Number of Reviews')
	plt.title('Beer Review Lengths')
	plt.hist(vals, bins, range=[0, 1500])
	#plt.show()
	plt.savefig('review_len.png')