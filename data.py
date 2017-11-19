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
datafile.create_dataset('trainingmovie_norm', data=movie_norm)
