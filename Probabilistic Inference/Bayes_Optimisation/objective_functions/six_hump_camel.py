from typing import Union, Tuple

import numpy as np
from matplotlib import pyplot as plt

from objective_functions.abstract_objective_function import ObjectiveFunction
from objective_functions.parameter_category import TypeVariable


class SixHumpCamelObjectiveFunction(ObjectiveFunction):
    def evaluate_without_noise(self,
                               data_points: np.ndarray
                               ) -> Union[np.ndarray, float]:
        """
        Same as evaluate(data_points) but does not apply any additional noise to the results
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return:  a numpy array of dimension n x 1 representing all the evaluations for all the n elements.
        """
        x = data_points[:, 0]
        y = data_points[:, 1]

        x2 = x ** 2
        x4 = x ** 4
        y2 = y ** 2

        return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 \
               + x * y \
               + (-4.0 + 4.0 * y2) * y2

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

        return (
            ((-3., 3.), TypeVariable.REAL),
            ((-2., 2.), TypeVariable.REAL),
        )

    def plot(self, list_number_points_per_axis):
        mesh_grid = self.get_mesh_grid(list_number_points_per_axis)
        xx, yy = mesh_grid
        xx, yy = xx.flatten(), yy.flatten()
        data_points = np.asarray([
            [x, y]
            for y in yy
            for x in xx
        ])

        evaluations = self.evaluate(data_points)
        evaluations = evaluations.reshape((xx.size, yy.size))
        levels = np.arange(-1.5, 10, 0.5)
        plt.contourf(mesh_grid[0].flatten(),
                     mesh_grid[1].flatten(),
                     evaluations,
                     levels=levels)
        plt.show()
