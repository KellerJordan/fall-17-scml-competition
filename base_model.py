# Template for the SMC competition for modeling neurons in the superior colliculus

import numpy as np
import h5py

# Please download the file SCNeuronModelCompetition.mat from here.
# https://github.com/santacruzml/fall-17-scml-competition/releases/download/0.0-data/SCNeuronModelCompetition.mat

datafile = h5py.File('SCNeuronModelCompetition.mat')
movie = datafile.get('trainingmovie_mini') # movie for training
frhist = datafile.get('FRhist_tr') # firing rate histograms

# a little normalization for the movie (assuming that the movie is 3D array)
def normalize(inputmovie):
    movie_mean = np.mean(inputmovie, axis=(0, 1, 2))
    movie_std = np.std(inputmovie, axis=(0, 1, 2))
    return (inputmovie - movie_mean) / movie_std

movie_norm = normalize(movie)


# here's the modeling part. I'll give just a starting point

import keras
from keras.layers import LSTM, Activation, Dense, BatchNormalization

# It makes a 3-layer LSTM network with batch normalization on each layer.
# No dropout, regularization, convolution structures are used.
# As you see in the summary, most parameters go to the first weight matrix.

movie_chunk_length = movie_norm.shape[1]
movie_pix = movie_norm.shape[2]
nHidden = 100
nLayer = 3
nSCNeu = frhist.shape[2]

model = keras.models.Sequential()
model.add(LSTM(nHidden, input_shape=(movie_chunk_length, movie_pix), return_sequences=True, implementation=2))

for _ in range(nLayer-1):
    model.add(BatchNormalization(momentum=0))
    model.add(Activation('relu'))
    model.add(LSTM(nHidden, return_sequences=True))
    
model.add(BatchNormalization(momentum=0))
model.add(Activation('linear'))
model.add(Dense(nSCNeu))
model.add(Activation('softplus'))
adamopt = keras.optimizers.Adam(lr = 0.001, decay = 1e-7)

# Please make sure to use Poisson likelihood function for the loss function
model.compile(optimizer=adamopt, loss='poisson')
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(movie_norm, frhist, epochs=200, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[early_stopping])


# check if it does a good job in the training dataset
%matplotlib inline
import matplotlib.pyplot as plt

output = model.predict(movie_norm)

for m in range(0, 48):
    n=31
    # plot the average of 6 trials of the same movie
    plt.plot(np.mean(frhist[(m*6):(m+1)*6, :, n], axis=(0)))
    
    # plot the output of the network
    plt.plot(output[m*6,:,n])
    plt.show()
    # last 10 movies should be the validation dataset
