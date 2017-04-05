import numpy as np

from tbnn import DataProcessor


"""
Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000,
there is a non-exclusive license for use of this work by or on behalf of the U.S. Government.
This software is distributed under the BSD-3-Clause license.
"""


class PlasticityDataProcessor(DataProcessor):
    """
    Inherits from DataProcessor class.  This class is specific to processing cubic crystal data to predict
    the stress tensor based on the deformation gradient tensor.
    """

    def calc_scalar_basis(self, A, B, is_train=False, cap=10.0, is_scale=True):
        """
        This returns a set of normalized scalar invariants for two symmetric tensors:
        Tr(A), Tr(AA), Tr(AAA), Tr(B), Tr(BB), Tr(BBB), Tr(AB), Tr(ABB), Tr(BAA), Tr(AABB)
        Reference for scalar basis:
        Spencer, A.J.M., 1987. Isotropic polynomial invariants and tensor functions.
        In Applications of tensor functions in solid mechanics (pp. 141-169). Springer Vienna.

        :param A: first symmetric tensor (strain)
        :param B: second symmetric tensor (plastic strain)
        :param is_train: Determines whether normalization constants should be reset
                        --True if it is training, False if it is test set
        :param cap: Caps the max value of the invariants after first normalization pass
        :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3)
        >>> B[0, :, :] = np.eye(3)*2.0
        >>> pdp = PlasticityDataProcessor()
        >>> pdp.mu = 0
        >>> pdp.std = 0
        >>> scalar_basis = pdp.calc_scalar_basis(A, B, is_scale=False)
        >>> print scalar_basis
        [[  3.   3.   3.   6.  12.  24.   6.  12.   6.  12.]]
        """

        num_points = A.shape[0]
        num_invariants = 10  # For 2 symmetric tensors, there are 10 scalar invariants
        invariants = np.zeros((num_points, num_invariants))
        for i in xrange(num_points):
            invariants[i, 0] = np.trace(A[i, :, :])
            invariants[i, 1] = np.trace(np.dot(A[i, :, :], A[i, :, :]))
            invariants[i, 2] = np.trace(np.dot(np.dot(A[i, :, :], A[i, :, :]), A[i, :, :]))
            invariants[i, 3] = np.trace(B[i, :, :])
            invariants[i, 4] = np.trace(np.dot(B[i, :, :], B[i, :, :]))
            invariants[i, 5] = np.trace(np.dot(np.dot(B[i, :, :], B[i, :, :]),
                                               B[i, :, :]))
            invariants[i, 6] = np.trace(np.dot(A[i, :, :], B[i, :, :]))
            invariants[i, 7] = np.trace(np.dot(np.dot(A[i, :, :], B[i, :, :]),
                                               B[i, :, :]))
            invariants[i, 8] = np.trace(np.dot(np.dot(B[i, :, :], A[i, :, :]), A[i, :, :]))
            invariants[i, 9] = np.trace(np.dot(np.dot(A[i, :, :], A[i, :, :]),
                                               np.dot(B[i, :, :], B[i, :, :])))

        # Renormalize invariants using mean and standard deviation:
        if is_scale:
            if self.mu is None or self.std is None:
                is_train = True

            if is_train:
                self.mu = np.zeros((num_invariants, 2))
                self.std = np.zeros((num_invariants, 2))
                self.mu[:, 0] = np.mean(invariants, axis=0)
                self.std[:, 0] = np.std(invariants, axis=0)
                self.std[self.std[:, 0] == 0, 0] = 1.0
            invariants = (invariants - self.mu[:, 0]) / self.std[:, 0]

            # Cap the data (useful for noisy data)
            invariants[invariants > cap] = cap
            invariants[invariants < -cap] = -cap

            # Now that we've capped, renormalize a second time
            invariants = invariants * self.std[:, 0] + self.mu[:, 0]
            if is_train:
                self.mu[:, 1] = np.mean(invariants, axis=0)
                self.std[:, 1] = np.std(invariants, axis=0)
                self.std[self.std[:, 1] == 0, 1] = 1.0
            invariants = (invariants - self.mu[:, 1]) / self.std[:, 1]

        return invariants

    def calc_tensor_basis(self, A, B, is_scale=True):
        """
        This returns the tensor basis for two symmetric tensors:
        A, AA, AAA, B, BB, BBB, AB, AAB, BBA, AABB
        Reference for tensor basis
        Smith, G.F., 1965. On isotropic integrity bases.
                 Archive for rational mechanics and analysis, 18(4), pp.282-292.

        :param A: first symmetric tensor (strain)
        :param B: second symmetric tensor (plastic strain)
        :return: T_flat: number of points X number of tensors in basis X 9
        >>> A = np.zeros((1, 3, 3))
        >>> B = np.zeros((1, 3, 3))
        >>> A[0, :, :] = np.eye(3)
        >>> B[0, :, :] = np.eye(3)*2.0
        >>> pdp = PlasticityDataProcessor()
        >>> tb = pdp.calc_tensor_basis(A, B, is_scale=False)
        >>> print tb[0, :, :]
        [[ 1.  0.  0.  0.  1.  0.  0.  0.  1.]
         [ 2.  0.  0.  0.  2.  0.  0.  0.  2.]
         [ 1.  0.  0.  0.  1.  0.  0.  0.  1.]
         [ 4.  0.  0.  0.  4.  0.  0.  0.  4.]
         [ 2.  0.  0.  0.  2.  0.  0.  0.  2.]
         [ 1.  0.  0.  0.  1.  0.  0.  0.  1.]
         [ 8.  0.  0.  0.  8.  0.  0.  0.  8.]
         [ 2.  0.  0.  0.  2.  0.  0.  0.  2.]
         [ 4.  0.  0.  0.  4.  0.  0.  0.  4.]
         [ 4.  0.  0.  0.  4.  0.  0.  0.  4.]]
        """

        num_points = A.shape[0]
        num_tensor_basis = 10  # For 2 symmetric tensors, there are 10 tensors in the basis
        T = np.zeros((num_points, num_tensor_basis, 3, 3))

        for i in range(num_points):
            aij = A[i, :, :]
            bij = B[i, :, :]
            T[i, 0, :, :] = aij
            T[i, 1, :, :] = bij
            T[i, 2, :, :] = np.dot(aij, aij)
            T[i, 3, :, :] = np.dot(bij, bij)
            T[i, 4, :, :] = np.dot(aij, bij)
            T[i, 5, :, :] = np.dot(aij, T[i, 2, :, :])
            T[i, 6, :, :] = np.dot(bij, T[i, 3, :, :])
            T[i, 7, :, :] = np.dot(aij, T[i, 4, :, :])
            T[i, 8, :, :] = np.dot(T[i, 3, :, :], aij)
            T[i, 9, :, :] = np.dot(T[i, 2, :, :], T[i, 3, :, :])

        # Scale down to promote convergence
        # Assumes magnitude of A is approximately sqrt(10)
        if is_scale:
            scale_factor = [3.1, 3.1, 10, 10, 10, 31, 31, 31, 31, 100]
            for i in range(num_tensor_basis):
                T[:, i, :, :] /= scale_factor[i]

        # Flatten:
        T_flat = np.zeros((num_points, num_tensor_basis, 9))
        for i in range(3):
            for j in range(3):
                T_flat[:, :, 3*i+j] = T[:, :, i, j]
        return T_flat

    def calc_output(self, plastic_strain_dot, strain_dot):
        """
        Flattens and normalizes the plastic strain time derivative tensor.
        Normalize plastic strain time derivative by dividing by the strain time derivative magnitude
        :param plastic_strain_dot: plastic strain time derivative
        :param strain_dot:  strain time derivative
        :return: plastic_strain_dot_flat: the flattened tensor represntation of the normalized plastic strain
        """

        # Calculate Frobenius norm of elastic strain time derivative tensor
        strain_dot_mag = np.sqrt(np.mean(np.mean(np.square(strain_dot), axis=2), axis=1))

        # To avoid singularities, if strain rate magnitude is zero, set it equal to one
        strain_dot_mag[strain_dot_mag == 0] = 1

        # Non-dimensionalize by the elastic strain time derivative tensor
        for i in range(3):
            for j in range(3):
                plastic_strain_dot[:, i, j] /= strain_dot_mag

        # Flatten
        num_points = plastic_strain_dot.shape[0]
        plastic_strain_dot_flat = np.zeros((num_points, 9))
        for i in range(3):
            for j in range(3):
                plastic_strain_dot_flat[:, 3*i+j] = plastic_strain_dot[:, i, j]
        return plastic_strain_dot_flat
