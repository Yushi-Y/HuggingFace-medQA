import numpy as np
import copy

import objective_functions.abstract_objective_function
import objective_functions.six_hump_camel
from acquisition_functions.abstract_acquisition_function import AcquisitionFunction
from gaussian_process import GaussianProcess
from kernels.abstract_kernel import Kernel


class BayesianOptimisation(object):

    def __init__(self,
                 kernel: Kernel,
                 objective_function: objective_functions.abstract_objective_function.ObjectiveFunction,
                 acquisition_function: AcquisitionFunction,
                 ):
        """
        :param kernel: Kernel object used by the gaussian process to perform a regression.
        :param objective_function: ObjectiveFunction object which we will try to minimise
        :param acquisition_function: AcquisitionFunction object
        """
        self._initial_kernel = copy.deepcopy(kernel)
        self._gaussian_process = GaussianProcess(kernel)
        self._objective_function = objective_function
        self._acquisition_function = acquisition_function

    def _initialise_gaussian_process(self,
                                     array_initial_dataset: np.ndarray,
                                     array_initial_objective_function_values: np.ndarray
                                     ) -> None:
        """
        Initialise the gaussian process with its initial dataset
        :param array_initial_dataset: array representing all the data points used to calculate the posterior mean and variance of the GP.
        Its dimension is n x l, there are:
        - n elements in the dataset. Each row corresponds to a data point x_i (with 1<=i<=n), at which the objective function can be evaluated
        - each one of them is of dimension l (representing the number of variables required by the objective function)
        :param array_initial_objective_function_values: array of the evaluations for all the elements in array_dataset. Its shape is hence n x 1 (it's a column vector)
        """

        self._gaussian_process.initialise_dataset(array_initial_dataset, array_initial_objective_function_values)

    def run(self,
            number_steps: int,
            array_initial_dataset: np.ndarray,
            array_initial_objective_function_values: np.ndarray,
            ) -> None:
        """
        Generator that performs a bayesian optimisation

        This method is a generator: at every step, it yields a tuple containing 3 elements:
        - the current up-to-date gaussian process
        - the acquisition function
        - the last computed argmax of the acquisition function.

        Hence, in order to use this method, you need to put it in a for loop,
            for gp, af, arg_max in bo.run(): # Here, bo is a BayesianOptimisation object
                # some code here


        :param number_steps: number of steps to execute in the Bayesian Optimisation procedure.

        :param array_initial_dataset: array_initial_dataset: array representing all the data points used to calculate the posterior mean and variance of the GP.
        Its dimension is n x l, there are:
        - n elements in the dataset. Each row corresponds to a data point x_i (with 1<=i<=n), at which the objective function can be evaluated
        - each one of them is of dimension l (representing the number of variables required by the objective function)

        :param array_initial_objective_function_values: array of the evaluations for all the elements in array_dataset. Its shape is hence n x 1 (it's a column vector)
        """

        print(f"Step {0}/{number_steps} - Initialise Gaussian Process for Provided Dataset")
        self._initialise_gaussian_process(array_initial_dataset,
                                          array_initial_objective_function_values)
        arg_max_acquisition_function = self.compute_arg_max_acquisition_function()

        for index_step in range(number_steps):
            print(f"Step {index_step}/{number_steps} - Evaluating Objective Function at position {arg_max_acquisition_function.tolist()}")
            arg_max_acquisition_function = self._bayesian_optimisation_step(arg_max_acquisition_function)

            # The yield keyword makes the method behave like a generator
            yield self._gaussian_process, self._acquisition_function, arg_max_acquisition_function

    def _bayesian_optimisation_step(self,
                                    arg_max_acquisition_function: np.ndarray
                                    ) -> np.ndarray:
        """
        :param arg_max_acquisition_function: the previously computed argmax of the acquisition function
        :return: the next computed arg_max of the acquisition function after having updated the Gaussian Process
        """
        objective_function_value = self._objective_function(arg_max_acquisition_function)
        self._gaussian_process.add_data_point(arg_max_acquisition_function, objective_function_value)

        self.reinitialise_kernel()
        self._gaussian_process.optimise_parameters()

        arg_max_acquisition_function = self.compute_arg_max_acquisition_function()

        return arg_max_acquisition_function

    def get_best_data_point(self) -> np.ndarray:
        index_best_data_point = np.argmin(self._gaussian_process.array_objective_function_values)
        return self._gaussian_process.array_dataset[index_best_data_point]

    def compute_arg_max_acquisition_function(self) -> np.ndarray:
        return self._acquisition_function.compute_arg_max(
            gaussian_process=self._gaussian_process,
            objective_function=self._objective_function
        )

    def reinitialise_kernel(self) -> None:
        self._gaussian_process.set_kernel_parameters(self._initial_kernel.log_amplitude,
                                                     self._initial_kernel.log_length_scale,
                                                     self._initial_kernel.log_noise_scale)
