import pandas as pd

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

		with open('beers.csv', 'w') as out:
		    df.to_csv(out)
		print('Total number of categories: ' + str(len(categories)))


main()