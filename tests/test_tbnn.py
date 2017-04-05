###############################################################
#
# Copyright 2017 Sandia Corporation. Under the terms of
# Contract DE-AC04-94AL85000 with Sandia Corporation, the
# U.S. Government retains certain rights in this software.
# This software is distributed under the BSD-3-Clause license.
#
##############################################################

from unittest import TestCase
from numpy.random import rand

import tbnn

###################################################
# Tests the consistency and basic functionality of
# the TBNN on a random set of data.  Does not
# require the pre-processing methods.
###################################################


class TestTbnn(TestCase):
    def test_tbnn_can_run(self):
        # parameters
        num_points = 100     # number of data points in the training set
        num_test = 50        # number of data points in the tests set
        num_features = 6     # number of input features
        num_basis = 3        # number of tensors in the basis
        basis_dim = 3        # dimensionality of the basis

        # number of elements will be basis_dim X basis_dim
        basis_elem_size = basis_dim * basis_dim

        x = rand(num_points, num_features)                  # random input data
        y = rand(num_points, basis_elem_size)               # random output data
        tb = rand(num_points, num_basis, basis_elem_size)   # random tensor basis

        # create the network structure, use defaults
        # num basis and num inputs will be set by TBNN during fit
        my_network = tbnn.NetworkStructure()

        # create the TBNN
        my_tbnn = tbnn.TBNN(my_network)
        # uses defaults in fit
        my_tbnn.fit(x, tb, y, max_epochs=100)

        # predict the responses for additional data
        x_test = rand(num_test, num_features)
        y_test = rand(num_test, basis_elem_size)
        tb_test = rand(num_test, num_basis, basis_elem_size)
        y_predict = my_tbnn.predict(x_test, tb_test)

        # check the error
        rmse = my_tbnn.rmse_score(y_test, y_predict)
        print "RMSE of tests case:", rmse
        self.assertTrue(rmse > 0.0)


