import cPickle as pickle
import random

"""
Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000,
there is a non-exclusive license for use of this work by or on behalf of the U.S. Government.
This software is distributed under the BSD-3-Clause license.
"""


class DataProcessor:
    """
    Parent class for data processing
    """
    def __init__(self):
        self.mu = None
        self.std = None

    def calc_scalar_basis(self, input_tensors, is_train=False, *args, **kwargs):
        if is_train is True or self.mu is None or self.std is None:
            print "Re-setting normalization constants"

    def calc_tensor_basis(self, input_tensors, *args, **kwargs):
        pass

    def calc_output(self, outputs, *args, **kwargs):
        return outputs

    @staticmethod
    def train_test_split(inputs, tb, outputs, fraction=0.8, randomize=True, seed=None):
        """
        Split inputs and outputs into training and validation set
        :param inputs: scalar invariants
        :param tb: tensor basis
        :param outputs: outputs
        :param fraction: fraction to use for training data
        :param randomize: if True, randomly shuffles data along first axis before splitting it
        :return:
        """
        num_points = inputs.shape[0]
        assert 0 <= fraction <= 1, "fraction must be a real number between 0 and 1"
        num_train = int(fraction*num_points)
        idx = range(num_points)
        if randomize:
            if seed:
                random.seed(seed)
            random.shuffle(idx)
        train_idx = idx[:num_train]
        test_idx = idx[num_train:]
        return inputs[train_idx, :], tb[train_idx, :, :], outputs[train_idx, :], \
               inputs[test_idx, :], tb[test_idx, :, :], outputs[test_idx, :]

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))
