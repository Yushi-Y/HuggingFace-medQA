import numpy as np
from scipy.spatial import distance_matrix

from kernels.abstract_kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self,
                 log_amplitude: float,
                 log_length_scale: float,
                 log_noise_scale: float,
                 ):
        super(GaussianKernel, self).__init__(log_amplitude,
                                             log_length_scale,
                                             log_noise_scale,
                                             )

    def get_covariance_matrix(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              ) -> np.ndarray:
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
        
        # Compute the squared euclidian distances between all the row vectors of X and Y
        # squared_distances = np.array([[np.linalg.norm(X[p,:]-Y[q,:])**2 for p in range(n1)] for q in range(n2)])
        # squared_distances.reshape((n1,n2))
        squared_distances = distance_matrix(X, Y)**2
        cov_matrix = amplitude_squared * np.exp((-1/2)/(ls**2)*squared_distances)
        
        return cov_matrix

    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ) -> np.ndarray:
        return self.get_covariance_matrix(X, Y)

