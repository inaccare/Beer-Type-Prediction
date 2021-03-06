import pandas as pd
import re
import sys
import time
import math
import numpy as np
np.set_printoptions(threshold=np.nan)
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import json
import nltk
import multiprocessing
import gensim
from collections import defaultdict
import codecs


"""
This model's architecture is as follows: product descriptions go into an LSTM layer, output of LSTM goes
through a 6-layer FC network. Output is a softmax vector of length 89, corresponding to 89 distinct 
Beer types.
"""

w2v = None
w2vFile = sys.argv[3]
w2v = gensim.models.word2vec.Word2Vec.load(w2vFile)

wordToIndex = dict()
indexToEmb = dict()
indexToWord = dict()
count = 1
wordToIndex['padding'] = 0
indexToWord[0] = 'padding'
indexToEmb[0] = np.random.randn(100)
for w in w2v.wv.vocab:
    indexToWord[count] = w
    wordToIndex[w] = count
    indexToEmb[count] = w2v.wv[w]
    count = count + 1

def main():
    # Usage is as follows: python model.py <train>.csv <dev>.csv <glove file> 
    X_test = []
    Y_test = []
    testCSV = None
    trainCSV = None
    devCSV = None
    if (len(sys.argv) >=2):
        trainCSV = sys.argv[1]
    if (len(sys.argv) >= 4):
        devCSV = sys.argv[2]
    lr, ne, bs, tx = None, None, None, 400
    if (len(sys.argv) >= 5):
        lr, ne, bs, tx = getHyperparamsFromJSON(str(sys.argv[4]))

    styleToIndex, indexToStyle = buildStyleIndices()
    trainDF = pd.read_csv(trainCSV, header = 0, encoding='utf-8', dtype={'text': str})
    rand_arr = np.random.rand(len(trainDF))
    msk_train = rand_arr < 0.01
    trainDF = trainDF[msk_train]
    # For each entry in X_train, we have an array of length T_x with each entry
    # corresponding to an index into the word's w2v embedding
    X_train, Y_train = getReviewIndicesAndStyles(trainDF, tx, styleToIndex)
    devDF = pd.read_csv(devCSV, header = 0, encoding='utf-8')
    rand_arr = np.random.rand(len(devDF))
    msk_dev = rand_arr > .999
    devDF = devDF[msk_dev]
    X_dev, Y_dev = getReviewIndicesAndStyles(devDF, tx, styleToIndex)
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_dev shape: " + str(X_dev.shape))
    print ("Y_dev shape: " + str(Y_dev.shape))
    if (lr == None):
        model(X_train, Y_train, X_dev, Y_dev, styleDict=indexToStyle)
    else:
        model(X_train, Y_train, X_dev, Y_dev, learning_rate = lr, num_epochs = ne, mini_batch_size = bs, Tx = tx, styleDict=indexToStyle)

def buildStyleIndices():
    styleDict = {}
    indexToStyleDict = {}
    df = pd.read_csv("beers_train.csv", header = 0, encoding='utf-8')
    styles = df['style'].unique()[:-1]
    for i, s in enumerate(styles):
        if str(s) not in styleDict:
            styleDict[str(s)] = i
            indexToStyleDict[i] = str(s)
    return styleDict, indexToStyleDict

def cleanStyles(styles):
    new_styles = []
    for s in styles:
        correct_string = s.replace('\xc3\xa9', 'e').replace('\xc3\xb6', 'o').replace('&#40;IPA&#41;', '').replace('&#40;Witbier&#41;', '').replace('\xc3\xa4', 'a').replace('\xc3\xa8', 'e').strip()
        new_styles.append(correct_string)
    return new_styles

def getReviewIndicesAndStyles(df, T_x, styleToIndex):
    X = []
    Y = []
    numBuckets = 89
    for i, row in df.iterrows():#len(df['item_description'])):
        if (pd.isnull(row['text']) == False): # Checks for Nan descriptions
            X.append(getIndexArrForSentence(row['text'], T_x))
        else:
            X.append(np.zeros(T_x))
        Y.append(OneHot(styleToIndex[str(row['style'])], 89))
    Y = np.array(Y).T
    X = np.array(X).T
    return X, Y

def getIndexArrForSentence(sentence, T_x):
    arr = np.zeros(T_x)
    words = nltk.word_tokenize(sentence.lower())
    count = 0
    for w in words:
        # Only looking at first 400 words!
        if (count == T_x):
            break
        if w in w2v.wv.vocab:
            arr[count] = wordToIndex[w]
            count = count + 1
    return arr

def getSentenceForIndexArr(arr):
    sentence = []
    for i in arr:
        sentence.append(indexToWord[int(i)])
    return sentence

def OneHot(bucket, numBuckets):
    """
    Creates onehot vector for our Y output
    Arguments:
    bucket -- index of correct bucket for example in Y
    numBuckets -- number of buckets used to split of the prices of objects
    Returns:
    arr -- onehot array
    """
    arr = np.zeros(numBuckets)
    arr[bucket] = 1
    return arr

# ==========
#   MODEL
# ==========
def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.01, num_epochs = 3,
        mini_batch_size = 128, Tx = 400, display_step = 1, n_hidden = 64, styleDict={}):
    # Shape of X: (m, Tx, n_x)??? Emmie please check this
    # Shape of Y: (n_y, m)
    print ("Model has following hyperparameters: learning rate: " + str(learning_rate) + ", num_epochs: " + str(num_epochs) + ", mini_batch_size: " \
        + str(mini_batch_size) + ", Tx: "+ str(Tx) + ".")

    # hidden layer num of features
    n_y = 89 # num categories
    n_x = 100 # w2v length

    # tf Graph input
    X = tf.placeholder("float", [None, Tx, n_x])
    Y = tf.placeholder("float", [n_y, None])
    # A placeholder for indicating each sequence length
    #Tx = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        # 'out': tf.Variable(tf.random_normal([n_hidden, n_y]))
        'W_1' : tf.get_variable('W_1',[n_hidden,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_2' : tf.get_variable('W_2',[n_hidden,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_out' : tf.get_variable('W_out',[n_hidden, n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_f1' : tf.get_variable('W_f1',[n_hidden, n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_f2' : tf.get_variable('W_f2',[n_hidden,n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1)),
        'W_fout' : tf.get_variable('W_fout',[n_hidden,n_y], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    }
    biases = {
        # 'out': tf.Variable(tf.random_normal([n_y]))
        'b_1' : tf.get_variable('b_1',[n_hidden], initializer = tf.zeros_initializer()),
        'b_2' : tf.get_variable('b_2',[n_hidden], initializer = tf.zeros_initializer()),
        'b_out' : tf.get_variable('b_out',[n_hidden], initializer = tf.zeros_initializer()),
        'b_f1' : tf.get_variable('b_f1',[n_hidden], initializer = tf.zeros_initializer()),
        'b_f2' : tf.get_variable('b_f2',[n_hidden], initializer = tf.zeros_initializer()),
        'b_fout' : tf.get_variable('b_fout',[n_y], initializer = tf.zeros_initializer())
    }

    lstm_output = dynamicRNN(X, Tx, weights, biases, n_x, n_hidden)
    tf.nn.dropout(lstm_output, 0.1)
    U = tf.get_variable("U", shape=[n_hidden * 2, n_y], initializer=tf.contrib.layers.xavier_initializer())
    b_last = tf.get_variable("b_last", shape=[1, n_y], initializer=tf.zeros_initializer())
    pred = tf.matmul(lstm_output, U) + b_last
#    pred = tf.Print(pred, [pred], message="This is pred: ", summarize=88)
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = tf.transpose(Y)))

    # derivative wrt to each embedding
    dc_de = tf.gradients(cost, X)
    sc = tf.linalg.norm(dc_de[0], axis=2)


    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Evaluate model
    predicted_vals = tf.argmax(pred, 1)
    actual_vals = tf.argmax(tf.transpose(Y), 1)

    correct_pred = tf.equal(predicted_vals, actual_vals) #Argmax over columns
    num_correct = tf.reduce_sum(tf.cast(correct_pred, tf.float32), name = "num_correct")

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Initialize the saver
    saver = tf.train.Saver()

    m = Y_train.shape[1]
    num_minibatches = int(math.floor(m/mini_batch_size))
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for step in range(1, num_epochs + 1):
            epoch_cost =0
            tot_num_correct = 0
            # extract each miniminibatch_X, miniBatch_Y at each
            #make minimatches here (randomly shuffling across m)
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = mini_batch_size, seed = 0)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                # Expand mininminibatch_X 
                minibatch_X = miniBatchIndicesToEmbedding(minibatch_X, Tx)# print ("Shape of minibatch_X is " + str(minibatch_X.shape))
                sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y})
                mini_num_correct, loss = sess.run([num_correct, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost = epoch_cost + loss
                tot_num_correct = tot_num_correct + mini_num_correct
                                               # Tx: Tx})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                                                    #Tx: Tx})
                print("Epoch " + str(step) + ", Cost= " + \
                      "{:.6f}".format(epoch_cost/num_minibatches) + ", Training Accuracy= " + \
                      "{:.5f}".format(float(tot_num_correct/m)))

        print("Optimization Finished!")
        train_num_correct = 0
        minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = mini_batch_size, seed = 0)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            minibatch_X = miniBatchIndicesToEmbedding(minibatch_X, Tx)
            num_correct_mb, loss = sess.run([num_correct, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            train_num_correct = train_num_correct + num_correct_mb
        print("Accuracy for train set: "+ str(train_num_correct/X_train.shape[1]))

        dev_num_correct = 0
        minibatches = random_mini_batches(X_dev, Y_dev, mini_batch_size = mini_batch_size, seed = 0)

        f = open('saliency.json', 'w')
        f.write('[\n')
        for minibatch in minibatches:
            (minibatch_X_ind, minibatch_Y) = minibatch
            minibatch_X = miniBatchIndicesToEmbedding(minibatch_X_ind, Tx)
            minibatch_sent = []
            for m in minibatch_X_ind.T:
                minibatch_sent.append(getSentenceForIndexArr(m))
            num_correct_mb, pr, ac, loss, saliency = sess.run([num_correct, predicted_vals, actual_vals, cost, sc], feed_dict={X: minibatch_X, Y: minibatch_Y})
            predicted = []
            actual = []
            for i in range(pr.shape[0]):
                predicted.append(styleDict[pr[i]])
                actual.append(styleDict[ac[i]])
            predicted = cleanStyles(predicted)
            actual = cleanStyles(actual)
            saliency_chart = list(zip(minibatch_sent, saliency))
            zipped = []
            for ex in saliency_chart:
                zipped.append(list(zip(ex[0], ex[1].tolist())))
            for i in range(len(predicted)):
                f.write('\t{\n')
                f.write('\t\t\"predicted\": \"' + predicted[i] + '\",\n')
                f.write('\t\t\"actual\": \"' + actual[i] + '\",\n')
                f.write('\t\t\"saliency\": ' + json.dumps(zipped[i]) + '\n')
                f.write('\t},\n')

            dev_num_correct = dev_num_correct + num_correct_mb
        f.write(']\n')
        f.close()


        print("Accuracy for dev set: "+ str(dev_num_correct/X_dev.shape[1]))
        saver.save(sess, './LSTM_model')
        sess.close()
        # # Calculate accuracy
        # test_data = testset.data
        # test_label = testset.labels
        # test_Tx = testset.Tx
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={X: test_data, Y: test_label,
        #                                   Tx: test_Tx}))

def dynamicRNN(X, Tx, weights, biases, n_x, n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (m, Tx, n_x)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input); or Tx tensors of shape (m, n_x)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # X = tf.unstack(X, Tx, 1) #Unstack to be (None, 100) vectors
    # Define a lstm cell with tensorflow
    lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    # Z_out, c = tf.contrib.rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
    _, output = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, X, dtype=tf.float32)
    output = (output[0][1], output[1][1])
    lstm_output = tf.concat(output, 1)
                                #sequence_length=Tx)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'Z_out' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    # Z_out = tf.stack(Z_out)
    # Z_out = tf.transpose(Z_out, [1, 0, 2])

    # # Hack to build the indexing and retrieve the right output.
    # batch_size = tf.shape(Z_out)[0]
    # # Start indices for each sample
    # index = tf.range(0, batch_size) * Tx + (Tx - 1)
    # # Indexing
    # Z_out = tf.gather(tf.reshape(Z_out, [-1, n_hidden]), index)
    # # Deepen LSTM network with fully connected
    # Z_out = tf.matmul(Z_out, weights['W_1']) + biases['b_1']
    # Z_out = tf.nn.relu(Z_out)
    # dropout = tf.layers.dropout(inputs=Z_out, rate=0.1) # Dropout 10% of units
    # Z_out = tf.matmul(Z_out, weights['W_2']) + biases['b_2']
    # Z_out = tf.nn.relu(Z_out)
    # dropout = tf.layers.dropout(inputs=Z_out, rate=0.1)
    # Z_out = tf.matmul(dropout, weights['W_out']) + biases['b_out']
    # Z_out = tf.nn.relu(Z_out)     

    # # Deepen Full Network
    # Z_out = tf.matmul(Z_out, weights['W_f1']) + biases['b_f1']
    # Z_out = tf.nn.relu(Z_out)
    # Z_out = tf.layers.dropout(inputs=Z_out, rate=0.1) # Dropout 10% of units
    # Z_out = tf.matmul(Z_out, weights['W_f2']) + biases['b_f2']
    # Z_out = tf.nn.relu(Z_out)
    # Z_out = tf.layers.dropout(inputs=Z_out, rate=0.1)
    # Z_out = tf.matmul(Z_out, weights['W_fout']) + biases['b_fout']

    # Linear activation, using Z_out computed above
    return lstm_output


# =================================================================
#   Helper functions for reading in data/getting minibatches ready
# =================================================================
def getHyperparamsFromJSON(filename):
    parameters = None
    with open(filename, 'r') as fp:
        parameters = json.load(fp)
    return float(parameters['learning_rate']), int(parameters['num_epochs']), int(parameters['batch_size']), int(parameters['Tx'])

def miniBatchIndicesToEmbedding(minibatch_X, T_x):
    m = minibatch_X.shape[1]# Maximum number of time steps (zero-padded)
    n_x = 100 # Length of words2vec vector for each word
    newArr = np.zeros((m, T_x, n_x))
    for i in range(m): # Iterating through samples
        for j in range(T_x): # Iterating through words
            indexToW2v = 0
            indexToW2v = minibatch_X[j,i]
            if  indexToW2v != 0:
                newArr[i,j,:] = np.array(indexToEmb[indexToW2v])
    return newArr

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, Tx)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = Y.shape[1]                 # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)    X shape: (Tx, m)   Y shape: (n_y, m)
    permutation = list(np.random.permutation(m))
    print("shape X", X.shape)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))  # not sure why we need to reshape here

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

if __name__ == '__main__':
    main()