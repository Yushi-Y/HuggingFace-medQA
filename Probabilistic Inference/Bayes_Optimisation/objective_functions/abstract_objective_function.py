import abc
from operator import itemgetter
from typing import Union, Tuple

import numpy as np

from objective_functions.parameter_category import TypeVariable


class ObjectiveFunction(metaclass=abc.ABCMeta):
    def __init__(self,
                 additional_gaussian_noise_std: float = 0.,
                 ):
        self._additional_gaussian_noise_std = additional_gaussian_noise_std

    def __call__(self,
                 data_points: np.ndarray
                 ) -> Union[np.ndarray, float]:
        return self.evaluate(data_points)

    def evaluate(self,
                 data_points: np.ndarray,
                 ) -> np.ndarray:
        """
        Evaluates the objective function at all the data points and adds a gaussian noise to it.
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of dimension n x 1 representing all the evaluations for all the n elements.
        """

        data_points = data_points.copy()
        data_points = data_points.reshape((-1,
                                           len(self.dataset_bounds)))

        data_points = self.floor_integer_parameters(data_points)
        result_without_noise = self.evaluate_without_noise(data_points)

        return result_without_noise + \
               np.random.normal(loc=0.,
                                scale=self._additional_gaussian_noise_std,
                                size=result_without_noise.shape)

    @abc.abstractmethod
    def evaluate_without_noise(self,
                               data_points: np.ndarray,
                               ) -> Union[np.ndarray, float]:
        """
        Same as evaluate(data_points) but does not apply any additional noise to the results
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return:  a numpy array of dimension n x 1 representing all the evaluations for all the n elements.
        """
        pass

    @property
    @abc.abstractmethod
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
        pass

    @property
    def boundaries(self):
        return tuple(map(itemgetter(0), self.dataset_bounds))

    def get_random_initial_dataset(self,
                                   number_initial_points: int
                                   ) -> np.ndarray:
        """
        Generates a set of number_initial_points elements uniformly sampled according to the dataset bounds
        :param number_initial_points:
        :return: A numpy array of dimension number_initial_points x m where m is the number of variables of the
        objective function
        """

        array_initial_dataset = np.array([]).reshape((0, len(self.dataset_bounds)))
        boundaries = tuple(map(itemgetter(0), self.dataset_bounds))

        for _ in range(number_initial_points):
            random_point = np.random.uniform(*zip(*boundaries))
            array_initial_dataset = np.vstack((array_initial_dataset, random_point))

        return array_initial_dataset

    def get_array_indexes_integer_parameters(self) -> np.ndarray:
        """
        :return: an array containing the indexes of the variables which are of type: TypeVariable.INTEGER
        """
        list_indexes_integer_parameters = []

        for index, (_, type_variable) in enumerate(self.dataset_bounds):
            if type_variable == TypeVariable.INTEGER:
                list_indexes_integer_parameters.append(index)

        return np.asarray(list_indexes_integer_parameters)

    def floor_integer_parameters(self, data_points: np.ndarray) -> np.ndarray:
        """
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of dimension n x m in which all variables of type TypeVariable.INTEGER
        are converted to integers
        """
        array_indexes_integer_parameters = self.get_array_indexes_integer_parameters()
        if array_indexes_integer_parameters.size > 0:
            data_points[:, array_indexes_integer_parameters] = np.floor(data_points[:, array_indexes_integer_parameters])
        return data_points

    def get_mesh_grid(self, list_number_points_per_axis):
        list_grid_points = []
        for index_axis, ((x_min, x_max), _) in enumerate(self.dataset_bounds):
            list_grid_points.append(np.linspace(x_min, x_max, list_number_points_per_axis[index_axis]))
        return np.meshgrid(*list_grid_points, sparse=True)

    def plot(self, list_number_points_per_axis):
        pass
