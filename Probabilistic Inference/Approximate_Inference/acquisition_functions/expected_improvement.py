from typing import Union

import numpy as np
from scipy.stats import norm

from acquisition_functions.abstract_acquisition_function import AcquisitionFunction
from gaussian_process import GaussianProcess


class ExpectedImprovement(AcquisitionFunction):
    def _evaluate(self,
                  gaussian_process: GaussianProcess,
                  data_points: np.ndarray
                  ) -> np.ndarray:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of shape n x 1 (or a float) representing the estimation of the acquisition function at
        each point
        """

        array_objective_function_values = gaussian_process.array_objective_function_values
        best_objective_function_value = np.min(array_objective_function_values)

        mean_data_points, std_data_points = gaussian_process.get_gp_mean_std(data_points)
        mean_data_points = mean_data_points.reshape((-1, 1))
        std_data_points = std_data_points.reshape((-1, 1))

        gamma = (best_objective_function_value - mean_data_points) / std_data_points

        return std_data_points * (gamma * norm.cdf(gamma) + norm.pdf(gamma))



