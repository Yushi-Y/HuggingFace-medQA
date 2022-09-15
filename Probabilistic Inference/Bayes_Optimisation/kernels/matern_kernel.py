import numpy as np
import math
from scipy.spatial import distance_matrix

from kernels.abstract_kernel import Kernel


class MaternKernel(Kernel):
    def get_covariance_matrix(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X: numpy array of size n_1 x m for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        # n1 = X.shape[0]
        # n2 = Y.shape[0]
        
        amplitude_squared = self.amplitude_squared
        ls = self.length_scale
        
        # Compute the L2 distances between all the row vectors of X and Y
        # L2_distances = np.array([[np.linalg.norm(X[p,:]-Y[q,:]) for p in range(n1)] for q in range(n2)])
        # L2_distances.reshape((n1, n2))
        L2_distances = distance_matrix(X, Y)
        cov_matrix = amplitude_squared * (1+math.sqrt(3)/ls*L2_distances) * np.exp(-math.sqrt(3)/ls*L2_distances)
        
        return cov_matrix
