# Tensor Basis Neural Network

The Tensor Basis Neural Network (TBNN) package provides an interface for building, training, and
testing neural networks for systems that have known invariance properties.  This package is intended
as a useful reference and starting point for researchers hoping to develop data-driven constitutive models
for systems with known invariance properties.  It was developed at Sandia National Laboratories with funding from the Sandia Laboratory Directed Research and Development program.

## Copyright and License

Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software.  This software is distributed under the BSD-3-Clause license.

## Reference:

Ling, Julia, Andrew Kurzawski, and Jeremy Templeton. "Reynolds averaged turbulence modelling using deep neural networks
with embedded invariance." Journal of Fluid Mechanics 807 (2016): 155-166.

## Installation:

This package requires the following python packages:
* lasagne 0.2.dev1
* theano 0.7.0.dev or above
* matplotlib 1.5.1 or above
* numpy 1.6.2 or above
* scipy 0.11 or above

To install, type: python setup.py install
To do developers install, type: python setup.py develop
To test installation, type: python setup.py test

## Background Theory:

In many physical systems, there are known invariance properties.  For example, the equations of motion
obey Galilean invariance, which means that they are invariant to inertial coordinate transformations.
Many crystals obey crystal symmetry constraints.  As we build machine learning models for these systems,
it has been demonstrated that we can achieve better accuracy at lower computational cost by directly embedding
these constraints.

The TBNN architecture enables the direct embedding of tensor invariance properties into the neural network
architecture.  The TBNN architecture is based on concepts from representation theory and tensor invariance
theory.

Lets say that we want to build a model where the inputs are tensors A and B and vector v and the output is
a tensor Y.  Furthermore, we know that Y obeys a known invariance constraint.  This can be expressed as:

if Y = Y(A, B, v)

then Y(Q A Q^T, Q B Q^T, Q v) = Q Y(A, B, v) Q^T


In this notation, Q is any matrix from the specified invariance group.  Q^T is the transpose of Q.  The
basic idea is that if the inputs are transformed by some matrix from the invariance group, then the output
should be transformed correspondingly.

For an arbitrary set of vectors and tensors and a specified invariance condition, it is possible to
construct a finite tensor basis (S<sub>1</sub>, S<sub>2</sub>, ..., S<sub>n</sub>) and a finite scalar basis (c<sub>1</sub>, c<sub>2</sub>, ..., c<sub>m</sub>).  
We can then say that any function of those
vectors and tensors should sit on the tensor basis:

Y = sum<sub>i</sub> [f<sub>i</sub>(c<sub>1</sub>, ..., c<sub>m</sub>) S<sub>i</sub> ]

In other words, our output Y can be expressed as a linear combination of the tensor basis, where the coefficients are
given by unknown functions f_i of the scalar invariants.  The tensor basis can either be calculated afresh or
can often be looked up in the literature, where many such bases have been tabulated.

The Tensor Basis Neural Network uses this concept to ensure that the model obeys the specified invariance constraints.
It has two input layers.  The first input layer accepts the scalar invariant basis.  These inputs are then transformed
through multiple densely connected hidden layers.  The second input layer accepts the tensor basis.  The final hidden
layer is then pair-wise multiplied by this tensor basis input layer and summed to provide the output.  Thus,
the TBNN directly maps the tensor basis equation into the neural network architecture.



## Examples:

Two different examples are provided in the examples folder.  They demonstrate how to use the TBNN library
to build, train, and test neural networks.  There is one example from 
turbulence modeling and one from plasticity in solid mechanics.

They can be run as:
```sh
cd examples/turbulence
python turbulence_example_driver.py
```
or as:
```sh
cd examples/plasticity
python plasticity_example_driver.py
```
These are toy examples, and are meant to show how the TBNN code base can be utilized and how to set up a data processor class to calculate the tensor and scalar invariant bases.

