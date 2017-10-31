import numpy as np
import h5py

datafile = h5py.File('SCNeuronModelCompetition.mat')
movie = datafile.get('trainingmovie_mini') # movie for training
frhist = datafile.get('FRhist_tr') # firing rate histograms


import matplotlib.pyplot as plt

for m in range(0, 48):
    n = 31
    # plot the average of 6 trials of the same movie
    plt.plot(np.mean(frhist[(m*6):(m+1)*6, :, n], axis=(0)))
    # plot the output of the network
    plt.show()
    # last 10 movies should be the validation dataset
