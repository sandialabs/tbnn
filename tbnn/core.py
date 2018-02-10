import numpy as np
import lasagne
import theano.tensor as T
import theano
import time
import cPickle as pickle

"""
Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000,
there is a non-exclusive license for use of this work by or on behalf of the U.S. Government.
This software is distributed under the BSD-3-Clause license.


Please note that parts of the TBNN.fit() method were based on the Lasagne python package tutorial,
available at http://lasagne.readthedocs.io/en/latest/user/tutorial.html, license available at
https://github.com/Lasagne/Lasagne/blob/master/LICENSE.
"""


class NetworkStructure:
    """
    A class to define the layer structure for the neural network
    """
    def __init__(self):
        self.num_layers = 1  # Number of hidden layers
        self.num_nodes = 10  # Number of nodes per hidden layer
        self.num_inputs = None  # Number of scalar invariants
        self.num_tensor_basis = None  # Number of tensors in the tensor basis
        self.nonlinearity = "LeakyRectify" # non-linearity string conforming to lasagne.nonlinearities tags
        self.nonlinearity_keywords = {}
        self.nonlinearity_keywords["leakiness"] = "0.1" # Leakiness of leaky ReLU activation functions

    def set_num_layers(self, num_layers):
        self.num_layers = num_layers
        return self

    def set_num_nodes(self, num_nodes):
        self.num_nodes = num_nodes
        return self

    def set_num_inputs(self, num_inputs):
        self.num_inputs = num_inputs
        return self

    def set_num_tensor_basis(self, num_tensor_basis):
        self.num_tensor_basis = num_tensor_basis
        return self

    def set_nonlinearity(self, nonlinearity):
        self.nonlinearity = nonlinearity
        return self

    def clear_nonlinearity_keywords(self):
        self.nonlinearity_keywords = {}

    def set_nonlinearity_keyword(self, key, value):
        if type(key) is not str:
            raise TypeError("NetworkStructure::set_nonlinearity_keywords - The keyword must be a string")
        # all values are stored as strings for later python eval
        if type(value) is not str:
            value = str(value)
        self.nonlinearity_keywords[key] = value
        return self


class TensorLayer(lasagne.layers.MergeLayer):
    """
    Multiplicative tensor merge layer.
    """
    def __init__(self, incomings, **kwargs):
        super(TensorLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        output_shape = (None, 9)
        return output_shape

    def get_output_for(self, inputs, **kwargs):
        output = T.batched_tensordot(inputs[0], inputs[1], axes=[[1], [1]])

        return output


class TBNN:
    """
    Defines Tensor Basis Neural Network (TBNN)
    :param train_fraction: the fraction of data used for training the NN vs. validation, must be between 0 and 1
    :param print_freq: frequency with which diagnostic data will be printed to the screen, in epochs, must be > 0
    :param learning_rate_decay: the decay rate for the learning rate effecting convergence, must be between 0 and 1
    :param min_learning_rate: minimum learning rate floor the optimizer will not go below, must be greater than 0
    """
    def __init__(self, structure=None, train_fraction=0.9,
                 print_freq=100, learning_rate_decay=1., min_learning_rate=1.e-6):
        if structure is None:
            structure = NetworkStructure()
        self.structure = structure
        self.network = None
        self.train_fraction = train_fraction
        self.print_freq = print_freq
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate

    def set_train_fraction(self, train_fraction):
        self.train_fraction = train_fraction
        return self

    def set_print_freq(self, print_freq):
        self.print_freq = print_freq
        return self

    def set_learning_rate_decay(self, learning_rate_decay):
        self.learning_rate_decay = learning_rate_decay
        return self

    def set_min_learning_rate(self, min_learning_rate):
        self.min_learning_rate = min_learning_rate
        return self

    def _build_NN(self):
        """
        Builds a TBNN with the number of hidden layers and hidden nodes specified in self.structure
        Right now, it is hard coded to use a Leaky ReLU activation function for all hidden layers
        """
        # determine type of non-linearity first
        nonlinearity_string = list("lasagne.nonlinearities."+self.structure.nonlinearity)
        # check if upper to know if there are args and () or not
        if self.structure.nonlinearity[0].isupper():
            nonlinearity_string += list("(")
            # add keyword options
            for key in self.structure.nonlinearity_keywords:
                nonlinearity_string += list(key+"="+self.structure.nonlinearity_keywords[key]+",")
            if self.structure.nonlinearity_keywords:
                nonlinearity_string[-1] = ")"
            else:
                nonlinearity_string += list(")")
        nonlinearity = eval("".join(nonlinearity_string))
        
        input_x = T.dmatrix('input_x')
        input_tb = T.dtensor3('input_tb')
        input_layer = lasagne.layers.InputLayer(shape=(None, self.structure.num_inputs), input_var=input_x)
        hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=self.structure.num_nodes,
            nonlinearity=nonlinearity, W=lasagne.init.HeUniform(gain=np.sqrt(2.0)))
        for i in xrange(self.structure.num_layers - 1):
            hidden_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=self.structure.num_nodes,
                nonlinearity=nonlinearity, W=lasagne.init.HeUniform(gain=np.sqrt(2.0)))
        linear_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=self.structure.num_tensor_basis,
            nonlinearity=None, W=lasagne.init.HeUniform(gain=np.sqrt(2.0)))

        tensor_layer = lasagne.layers.InputLayer(shape=(None, self.structure.num_tensor_basis, 9), input_var=input_tb)
        merge_layer = TensorLayer([linear_layer, tensor_layer])
        self.network = merge_layer

    def _check_structure(self, x, tb, y):
        """
        Define number of inputs and tensors in tensor basis and check that they're consistent with
        the specified structure
        :param x: Matrix of input features.  Should be num_points X num_features numpy array
        :param tb: Matrix of tensor basis.  Should be num_points X num_tensor_basis X 9 numpy array
        :param y: Matrix of labels.  Should by num_points X 9 numpy array
        """

        # Check that the inputs, tensor basis array, and outputs all have same number of data points
        assert x.shape[0] == y.shape[0], "Mis-matched shapes between inputs and outputs"
        assert x.shape[0] == tb.shape[0], "Mis-matched shapes between inputs and tensors"

        # Define number of inputs and tensors in tensor basis and check that they're consistent with
        # the specified structure
        if self.structure.num_inputs is None:
            self.structure.set_num_inputs(x.shape[-1])
        else:
            if self.structure.num_inputs != x.shape[-1]:
                print "Mis-matched shapes between specified number of inputs and number of features in input array"
                raise Exception

        if self.structure.num_tensor_basis is None:
            self.structure.set_num_tensor_basis(tb.shape[1])
        else:
            if self.structure.num_tensor_basis != tb.shape[1]:
                print "Mis-matched shapes between specified number of tensors in \
                 tensor basis and number of tensors in tb"
                raise Exception

    def fit(self, scalar_basis, tensor_basis, labels, max_epochs=1000, min_epochs=0, init_learning_rate=0.01,
            interval=10, average_interval=10, loss=None, optimizer='adam'):
        """
        Fit the Tensor Basis Neural Network to the data.  Note: Parts of this method is based on the Lasagne tutorial
        available at http://lasagne.readthedocs.io/en/latest/user/tutorial.html.
        :param scalar_basis: Matrix of scalar basis features.  Should be num_points X num_features numpy array
        :param tensor_basis: Matrix of tensor basis.  Should be num_points X num_tensor_basis X 9 numpy array
        :param labels: Matrix of labels.  Should by num_points X 9 numpy array
        :param max_epochs: Maximum number of training epochs
        :param min_epochs: Minimum number of training epochs
        :param init_learning_rate: Initial learning rate to use
        :param interval: Frequency at which convergence criteria are checked
        :param average_interval: Number of intervals averaged over to determine if early stopping criteria should be triggered
        :param loss: If desired, can specify a loss function.  If none specified, the default is
                     a mean squared error loss function.
        :param optimizer: Can specify which optimizer to use.
                          Valid options include: "adam", "sgd", "rmsprop", "momentum"
        :return:
        """

        # Build the neural network
        self._check_structure(scalar_basis, tensor_basis, labels)
        self._build_NN()

        optimizer_function = {
            "adam": lasagne.updates.adam,
            "sgd": lasagne.updates.sgd,
            "momentum": lasagne.updates.momentum,
            "rmsprop": lasagne.updates.rmsprop}[optimizer]

        # Split into training and validation for early stopping
        num_points = scalar_basis.shape[0]
        num_train = int(self.train_fraction*num_points)
        x_train = scalar_basis[:num_train, :]
        y_train = labels[:num_train, :]
        tb_train = tensor_basis[:num_train, :, :]
        x_valid = scalar_basis[num_train:, :]
        y_valid = labels[num_train:, :]
        tb_valid = tensor_basis[num_train:, :, :]

        # Specify the loss function and the training parameters
        output_var = T.dmatrix('outputs')
        input_x = lasagne.layers.get_all_layers(self.network)[0].input_var
        input_tb = lasagne.layers.get_all_layers(self.network)[-2].input_var
        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        if loss is None:
            loss = lasagne.objectives.squared_error(prediction, output_var)
            loss = lasagne.objectives.aggregate(loss, mode='mean')
        valid_function = theano.function([input_x, input_tb, output_var], loss, on_unused_input='warn')
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        learning_rate = theano.shared(np.array(init_learning_rate, dtype=theano.config.floatX))
        learning_rate_decay = np.array(self.learning_rate_decay, dtype=theano.config.floatX)
        min_learning_rate = np.array(np.minimum(self.min_learning_rate, init_learning_rate), dtype=theano.config.floatX)
        updates = optimizer_function(loss, params, learning_rate=learning_rate)
        train_function = theano.function([input_x, input_tb, output_var], loss, updates=updates, on_unused_input='warn')

        epoch = 0
        keep_going = True
        batch_size = 1
        error = np.zeros((1,))

        def iterate_mini_batches(x1, x2, target, size, shuffle=False):
            """
            Select mini-batches
            :param x1: scalar basis
            :param x2: tensor basis
            :param target: targets
            :param size: size of batch
            :param shuffle: whether or not to shuffle points
            :return:
            """
            assert len(x1) == len(target)
            indices = np.arange(len(x1))
            if shuffle:
                np.random.shuffle(indices)
            for start_idx in range(0, len(x1) - size + 1, size):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + size]
                else:
                    excerpt = slice(start_idx, start_idx + size)
                yield x1[excerpt], x2[excerpt], target[excerpt]
        
        def check_convergence(error, max_epochs=30000, min_epochs=1000, interval=1, average_interval=10):
            """
            Check convergence criteria for early stopping and maximum number of epochs
            :param error:
            :param max_epochs: Maximum number of epochs to run (This over-rides min_epochs)
            :param min_epochs: Minimum number of epochs to run before stopping
            :param interval: How often convergence is checked
            :return:
            """

            # Calculate the true epoch number based on the interval and the number of entries in validation error array
            epoch = (error.shape[0] - 2) * interval + 1

            keep_going = True
            if epoch > min_epochs:
                # Implement an early stopping criterion to halt training if the validation error starts rising
                keep_going = np.mean(error[-average_interval:]) < np.mean(error[-2*average_interval:-average_interval])
            if epoch > max_epochs:
                keep_going = False
            return keep_going
        
        # Do stochastic gradient descent
        print("Epoch time training_loss validation_loss")
        while keep_going:
            start_time = time.time()
            learning_rate.set_value(np.maximum(learning_rate.get_value()*learning_rate_decay, min_learning_rate))

            # In each epoch, we do a full pass over the training data:
            train_error = 0
            train_batches = 0
            # Timer used to be here
            for batch in iterate_mini_batches(x_train, tb_train, y_train, size=batch_size, shuffle=True):
                inputs, input_tensors, targets = batch
                train_error += train_function(inputs, input_tensors, targets)
                train_batches += 1

            # Check for early stopping criteria
            if epoch % interval == 0:
                val_error = valid_function(x_valid, tb_valid, y_valid)
                error = np.hstack((error, val_error))
            keep_going = check_convergence(error, max_epochs=max_epochs, min_epochs=min_epochs, interval=interval, average_interval=average_interval)

            # Then we print the results for this epoch:
            if epoch % self.print_freq == 0:
                train_error = valid_function(x_train, tb_train, y_train)
#               print("Epoch {} took {:.3f}s".format(epoch + 1, time.time() - start_time))
#               print("  rmse training loss:\t\t{:.6f}".format(np.sqrt(train_error)))
#               print("  rmse validation loss:\t\t{:.6f}".format(np.sqrt(val_error)))
                print("{0:6d} {1:7.4f} {2:12.8f} {3:12.8f}".format(
                    epoch + 1, time.time() - start_time,
                    np.sqrt(train_error),np.sqrt(val_error)))

            epoch += 1

        print "Total number of epochs: ", epoch
        print "Final rmse validation error: ", np.sqrt(val_error)

    def predict(self, x, tb):
        """
        Make a prediction for a given set of input scalar invariants x and tensor basis tb
        :param x: scalar basis
        :param tb: tensor basis
        """
        input_x = lasagne.layers.get_all_layers(self.network)[0].input_var
        input_tb = lasagne.layers.get_all_layers(self.network)[-2].input_var
        prediction_function = theano.function([], lasagne.layers.get_output(self.network, deterministic=True),
                                     givens={input_x: x, input_tb: tb}, on_unused_input='warn')
        y_predicted = prediction_function()
        return y_predicted

    def predict_tensor_coefs(self, x, tb):
        """
        Make a prediction for a given set of input scalar invariants x and tensor basis tb
        :param x: scalar basis
        :param tb: tensor basis
        """
        input_x = lasagne.layers.get_all_layers(self.network)[0].input_var
        prediction_function = theano.function([], lasagne.layers.get_output(lasagne.layers.get_all_layers(self.network)[-3], deterministic=True),
                                     givens={input_x: x}, on_unused_input='warn')
        y_predicted = prediction_function()
        return y_predicted

    def rmse_score(self, y_true, y_predicted):
        """
        Calculate root mean squared error (RMSE)
        :param y_true: true value
        :param y_predicted: predicted value
        >>> tbnn = TBNN()
        >>> print tbnn.rmse_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 5.0, -1.0]))
        2.88675134595
        """
        assert y_true.shape == y_predicted.shape, "Shape mismatch"
        rmse = np.sqrt(np.mean(np.square(y_true-y_predicted)))
        return rmse

    def contour_plot(self, x_coords, y_coords, f):
        import matplotlib.pyplot as plt
        """
        Make a contour plot
        :param x_coords: x coordinates of points
        :param y_coords: y coordinates of points
        :param f: Function to use for coloring
        :return:
        """
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.gca()
        ax.set_aspect('equal')
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        triangles = tri.Triangulation(x_coords, y_coords)
        for i in range(9):
            sub = plt.subplot(3, 3, i+1)
            plt.tricontourf(triangles,  f[:, i], cmap='Spectral_r', extend='both')
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
               labelbottom='off', labelleft='off')

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))


