# -*- coding: utf-8 -*-

import cmath
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

def label_to_int(label):
    labels = ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
    for i,label_name in enumerate(labels):
        if(label == label_name):
            return i
    return -1

def load_dataset(pickle_path):
    # Data Preprocessing
    df = pd.read_pickle(pickle_path)
    df.iloc[:,:-1] = df.iloc[:,:-1].applymap(lambda cell: [np.real(cell), np.imag(cell)])
    df = df.sample(frac=1).reset_index(drop=True)
    X = np.zeros((df.iloc[:,:-1].shape[0],df.iloc[:,:-1].shape[1],2))
    for row_index,row in enumerate(df.iloc[:,:-1].values):
        for col_index,cell in enumerate(row):
            X[row_index][col_index] = cell
    y = np.array([])
    for label in df.iloc[:,-1].values:
        y = np.append(y,label_to_int(label))
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print("Network input dim: {}, X_train: {}, X_test: {}, y_train: {}, y_test: {}".format((df.shape[1]-1,2),X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    return X_train, X_test, y_train, y_test, (df.shape[1]-1,2)

def evaluate_model(X_train, X_test, y_train, y_test, input_shape):
    verbose, epochs, batch_size = 1, 150, 32

    # Initialising the CNN
    classifier = Sequential()
    classifier.add(Convolution1D(filters=8, kernel_size=1, input_shape=input_shape, activation='relu'))
    classifier.add(Convolution1D(filters=4, kernel_size=3, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(MaxPool1D(pool_size=2))
    classifier.add(Flatten())
    classifier.add(Dense(units=32, activation='relu'))
    classifier.add(Dense(units=5, activation='softmax'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit network
    classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = classifier.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return(accuracy)

def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# load data
	X_train, X_test, y_train, y_test, input_shape = load_dataset('./dataset/coeff_dataset.pkl')
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(X_train, X_test, y_train, y_test, input_shape)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment
run_experiment(1)
