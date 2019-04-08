import math

import numpy as np

'''
This file provides the data provider in experiments.
After construction, you can iterate it to get batches.
It shuffles records at the beginning of each epoch.
'''


class Provider:
    def __init__(self, source, resolution, continuous, batchsize, spec='cqt', phase=False):
        """
        :param source: one in train, test, valid
        :param resolution: resolution of spectrogram 1: original, 2: half etc.
        :param continuous: true for continuous dada, false for average
        :param batchsize: batch size
        """
        self.phase = phase
        self.group = source
        self.spec = spec
        self.continuous = continuous
        self.size = batchsize
        self.data = np.load('data/deam-{0}-{1}.npz'.format(resolution, source))

    def __iter__(self):
        amp = self.data['amp_' + self.spec]
        pha = self.data['pha_' + self.spec]

        if self.continuous:
            valence = self.data['valence_cont'] / 2
            arousal = self.data['arousal_cont'] / 2
        else:
            valence = (self.data['valence_mean'] - 5) / 8
            arousal = (self.data['arousal_mean'] - 5) / 8
        idx = np.random.permutation(valence.shape[0])
        amp = amp[idx]
        pha = pha[idx]
        valence = valence[idx]
        arousal = arousal[idx]
        start = 0
        end = min(start + self.size, valence.shape[0])
        while True:
            y = np.hstack((valence[start:end], arousal[start:end]))
            if self.phase:
                x = np.stack((i[start:end, :, :] for i in [amp, pha]), axis=1)
            else:
                x = np.expand_dims(amp[start:end, :, :], axis=1)
            yield x, y
            start = end
            end = min(start + self.size, valence.shape[0])
            if start == end:
                break

    def shape(self):
        return (2 if self.phase else 1, *self.data['amp_' + self.spec][0].shape)

    def __len__(self):
        return math.ceil(len(self.data['valence_mean']) / self.size)

    def total(self):
        return len(self.data['valence_mean'])
