import pandas as pd
import numpy as np

data_path = '../ratebeer.txt'

def main():
	with open(data_path, encoding='utf-8') as f:
		reviews = f.read().split('\n\n')
		parsed_data = []
		categories = set()
		num_reviews = len(reviews)
		for num, r in enumerate(reviews):
			if num % 10000 == 0:
				print(str(num) + '/' + str(num_reviews))
			datum = []
			feats = r.split('\n')
			for i, feat in enumerate(feats):
				values = feat.split(':')
				if len(values) < 2:
					value = 'N/A'
				else:
					value = ':'.join(values[1:]).strip()
				if i == 4:
					categories.add(value)
				datum.append(value)
			parsed_data.append(datum)

		columns = ['name', 'beerId', 'brewerId', 'ABV', 'style', 'appearance', 'aroma', 'palate', 'taste', 'overall', 'time', 'profileName', 'text']
		df = pd.DataFrame(parsed_data)
		df.columns = columns

		arr = np.random.rand(len(df))
		msk_train = arr < 0.9
		msk_dev = arr > 0.95
		msk_test = arr > 0.9
		train = df[msk_train]
		dev = df[msk_dev]
		test = df[msk_test & ~msk_dev]

		with open('beers_train.csv', 'w') as out:
		    train.to_csv(out)
		print('Total number of categories: ' + str(len(categories)))

		with open('beers_dev.csv', 'w') as out:
		    dev.to_csv(out)

		with open('beers_test.csv', 'w') as out:
		    test.to_csv(out)

main()