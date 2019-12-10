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

# Data Preprocessing
df = pd.read_pickle('./dataset/coeff_dataset.pkl')
df.iloc[:,:-1] = df.iloc[:,:-1].applymap(lambda cell: [np.real(cell), np.imag(cell)])
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Initialising the CNN
input_shape = (df.shape[1]-1,2)
classifier = Sequential()
classifier.add(Convolution1D(filters=2, kernel_size=1, input_shape=input_shape, activation='relu'))
classifier.add(Convolution1D(filters=1, kernel_size=3, activation='relu'))
classifier.add(MaxPool1D(pool_size=2))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])