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

def predictions_to_label(predictions):
    labels = ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
    labels_score = {}
    for i,prediction in enumerate(predictions[0]):
        label = labels[i]
        labels_score[label] = prediction
    return labels_score

def max_prediction(predictions):
    labels = ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
    max_pred_score = 0
    max_pred = 'no_hand'
    for i,prediction in enumerate(predictions[0]):
        if(prediction>max_pred_score):
            max_pred_score = prediction
            max_pred = labels[i]
    return max_pred
    
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

def save_model(model,fullpath):
    # serialize model to JSON
    model_json = model.to_json()
    with open(fullpath + ".json", "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

def save_weights(model,fullpath):
    # serialize weights to HDF5
    model.save_weights(fullpath + ".h5")
    print("Saved weights to disk")
 
def evaluate_model(X_train, X_test, y_train, y_test, input_shape,save_model_bool=False,save_weights_bool=False):
    verbose, epochs, batch_size = 1, 150, 32
    print(input_shape)
    
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
    classifier.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = classifier.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    if save_model_bool:
        save_model(classifier,"./networks/Conv8Conv4Drop05Dense32Quick")
    if save_weights_bool:
        save_weights(classifier, "./networks/Conv8Conv4Drop05Dense32E150B32Quick")
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
		score = evaluate_model(X_train, X_test, y_train, y_test, input_shape,True,True)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

 
# run the experiment
# run_experiment(1)

 

# # later...
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))