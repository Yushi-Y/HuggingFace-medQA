from typing import Union, Tuple

import numpy as np
from matplotlib import pyplot as plt

from objective_functions.abstract_objective_function import ObjectiveFunction
from objective_functions.parameter_category import TypeVariable


class Linear(ObjectiveFunction):
    def __init__(self,
                 additional_gaussian_noise_std: float = 0.,
                 ):
        super(Linear, self).__init__(additional_gaussian_noise_std)

    def evaluate_without_noise(self,
                               x: Union[np.ndarray, float],
                               ) -> Union[np.ndarray, float]:
        """
        Same as evaluate(data_points) but does not apply any additional noise to the results
        :param x: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of dimension n x 1 representing all the evaluations for all the n elements.
        """

        result = 0.5 * x + 1
        return result

    @property
    def dataset_bounds(self) -> Tuple[Tuple[Tuple[float, float],
                                            TypeVariable],
                                      ...]:
        """
        Defines the bounds and the types of variables for the objective function

        Example:
        if dataset_bounds is equal to
        (
        ((1, 2), TypeVariable.REAL),
        ((5, 10), TypeVariable.INTEGER),
        )
        then it means the objective function depends on 2 variables:
        - the first one is a real number between 1 and 2
        - the second one is an integer between 5 (included) and 10 (excluded)
        """

        return ((0., 10.), TypeVariable.REAL),

    def plot(self, list_number_points_per_axis):
        mesh_grid = self.get_mesh_grid(list_number_points_per_axis)

        evaluations = self.evaluate(mesh_grid[0])
        if not self._additional_gaussian_noise_std:
            plt.plot(mesh_grid[0].flatten(),
                     evaluations.flatten())
        else:
            plt.scatter(mesh_grid[0].flatten(),
                        evaluations.flatten())
        plt.show()