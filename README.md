# Beer-Type-Prediction

To read in the w2v to a dict:
w2v = None
w2vFile = sys.argv[3]
w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)
