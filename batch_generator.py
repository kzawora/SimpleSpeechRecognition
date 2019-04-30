import os
import numpy
import numpy as np
import librosa
import global_variables as gv
from random import shuffle


def mfcc_batch_generator(batch_size=10, height=gv.height, path=gv.pcm_path):
    batch_features = []
    labels = []
    names = []
    files = os.listdir(path)
    while True:
        print("loaded batch of %d files" % len(files))
        shuffle(files)
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            wave, sr = librosa.load(path + wav, mono=True, sr=None)
            label = dense_to_one_hot(int(wav[0]), 10)
            labels.append(label)
            names.append(wav)
            mfcc = librosa.feature.mfcc(wave, sr, n_fft=5000, hop_length=2500, n_mels=30)
            mfcc = np.pad(mfcc, ((0, 0), (0, height - len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                yield batch_features, labels, names  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch
                labels = []
                names = []


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return numpy.eye(num_classes)[labels_dense]
