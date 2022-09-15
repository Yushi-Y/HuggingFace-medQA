import abc
from operator import itemgetter
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

from gaussian_process import GaussianProcess
from objective_functions.abstract_objective_function import ObjectiveFunction


class AcquisitionFunction(metaclass=abc.ABCMeta):
    def compute_arg_max(self,
                        gaussian_process: GaussianProcess,
                        objective_function: ObjectiveFunction,
                        ) -> np.ndarray:
        """
        :param gaussian_process:
        :param objective_function:
        :return: The computed arg_min of the acquisition of the acquisition function according to:
        - the predictions (mean and std) of the gaussian process
        - the boundaries of the objective function
        The result is a flattened numpy array.
        """

        def evaluate_gaussian_process(data_point: np.ndarray) -> float:
            return -1 * self.evaluate(gaussian_process, objective_function, data_point)

        boundaries = tuple(map(itemgetter(0), objective_function.dataset_bounds))

        if len(boundaries) == 1:
            data_points, = objective_function.get_mesh_grid([100])
            data_points = data_points.reshape((-1, 1))
            evaluations = self.evaluate(gaussian_process, objective_function, data_points)
            return data_points[np.argmax(evaluations)]

        elif len(boundaries) == 2:
            data_points, yy = objective_function.get_mesh_grid([100, 100])
            data_points, yy = data_points.flatten(), yy.flatten()
            data_points = np.asarray([
                [x, y]
                for y in yy
                for x in data_points
            ])
            evaluations = self.evaluate(gaussian_process, objective_function, data_points)
            return data_points[np.argmax(evaluations)]
        else:

            initial_point = np.random.uniform(*zip(*boundaries))
            res = scipy.optimize.fmin_l_bfgs_b(func=evaluate_gaussian_process,
                                               x0=initial_point,
                                               bounds=boundaries,
                                               approx_grad=True)
            return res[0]

    def __call__(self,
                 gaussian_process: GaussianProcess,
                 objective_function: ObjectiveFunction,
                 data_points: np.ndarray
                 ) -> Union[float, np.ndarray]:
        return self.evaluate(gaussian_process, objective_function, data_points)

    def evaluate(self,
                 gaussian_process: GaussianProcess,
                 objective_function: ObjectiveFunction,
                 data_points: np.ndarray,
                 ) -> np.ndarray:
        """
        If some values in data_points are supposed to be integers, we convert them to integers here before calling
        the method _evaluate

        :param gaussian_process:
        :param objective_function:
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: estimates the value of the acquisition function for each element depending on:
        - the predictions of the gaussian process
        """
        data_points = data_points.copy()
        data_points = data_points.reshape((-1, len(objective_function.dataset_bounds)))

        # floor integers before evaluating
        data_points = objective_function.floor_integer_parameters(data_points)

        return self._evaluate(gaussian_process, data_points)

    @abc.abstractmethod
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

    def plot(self,
             gaussian_process: GaussianProcess,
             objective_function: ObjectiveFunction,
             last_evaluated_point: Union[float, np.ndarray]):
        number_dimensions = len(objective_function.dataset_bounds)

        if number_dimensions == 1:
            mesh_grid = objective_function.get_mesh_grid([100])

            evaluations = self.evaluate(gaussian_process, objective_function, mesh_grid[0])
            plt.plot(mesh_grid[0].flatten(),
                     evaluations.flatten())
            plt.scatter(gaussian_process.array_dataset.flatten(),
                        np.full_like(gaussian_process.array_dataset.flatten(), np.min(evaluations)),
                        color='b',
                        marker='x')
            plt.axvline(last_evaluated_point, color='r')

            plt.title("Acquisition Function")
            plt.show()
        elif number_dimensions == 2:
            mesh_grid = objective_function.get_mesh_grid([100, 100])
            xx, yy = mesh_grid
            xx, yy = xx.flatten(), yy.flatten()
            data_points = np.asarray([
                [x, y]
                for y in yy
                for x in xx
            ])

            evaluations = self.evaluate(gaussian_process, objective_function, data_points)
            evaluations = evaluations.reshape((xx.size, yy.size))

            contour_levels = [
                np.percentile(evaluations, k) for k in range(101)
            ]
            contour_levels = sorted(list(set(contour_levels)))
            # levels = np.arange(np.min(evaluations), np.min(evaluations) + 12, 0.5)
            # levels = np.hstack((levels, np.max(evaluations)))
            CS = plt.contourf(mesh_grid[0].flatten(),
                              mesh_grid[1].flatten(),
                              evaluations,
                              levels=contour_levels)
            plt.scatter(gaussian_process.array_dataset[:, 0].flatten(),
                        gaussian_process.array_dataset[:, 1].flatten(),
                        color='b',
                        marker='+')
            plt.scatter(last_evaluated_point[0],
                        last_evaluated_point[1],
                        color='r',
                        marker='x')

            plt.colorbar(CS)
            plt.title("Acquisition Function")
            plt.show()
