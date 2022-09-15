import jax.numpy as np
import jax.scipy
import numpy as onp
from jax import grad
from jax.config import config

from distribution_prediction.blackbox_vi.utils_plots import plot_vi_gp

config.update("jax_debug_nans", True)

from objective_functions.sin import LinearSin


def get_distances_array(X_1, X_2):
    """
    Compute the array of euclidian distances between points from two different sets: X_1 and X_2

    :param X_1: numpy array of size n_1 x m for which each row (x_1_i) is a data point at which the objective function can be evaluated
    :param X_2: numpy array of size n_2 x m for which each row (x_2_j) is a data point at which the objective function can be evaluated
    :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        the euclidian distance between x_1_i and x_2_j
            distances_array[i,j] = euclidian distance between x_1_i and x_2_j
    """
    distances_array = np.array([[np.linalg.norm(x_p - x_q) for x_q in X_2] for x_p in X_1])
    return distances_array


def get_cov_matrix_gaussian_linear(amplitude_gaussian_squared: float,
                                   length_scale: float,
                                   amplitude_linear_squared: float,
                                   offset_squared: float,
                                   c: float,
                                   X_1: np.ndarray,
                                   X_2: np.ndarray,
                                   distances_array: np.ndarray
                                   ) -> np.ndarray:
    """
    Calculates the Covariance Matrix according to a GaussianLinear Kernel
    k(x,y) = gaussian_kernel(x,y) + amplitude_linear_squared * (x - c).dot((y - c).T) + offset_squared

    :param amplitude_gaussian_squared: Parameter of the gaussian kernel
    :param length_scale: Parameter of the gaussian kernel
    :param amplitude_linear_squared: Parameter of the Linear kernel
    :param offset_squared: Parameter of the Linear kernel
    :param c: Parameter of the Linear kernel
    :param X_1: numpy array of size n_1 x m for which each row (x_1_i) is a data point at which the objective function can be evaluated
    :param X_2: numpy array of size n_2 x m for which each row (x_2_j) is a data point at which the objective function can be evaluated
    :param distances_array: array of distances between the points in X_1 and X_2:
        distances_array[i,j] = euclidian distance between x_1_i and x_2_j
    :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_1_i, x_2_j), where k represents the kernel used.
    """

    return (amplitude_gaussian_squared
            * np.exp((-1 / (2 * length_scale ** 2))
                     * (distances_array ** 2))
            ) + amplitude_linear_squared * (X_1 - c).dot(X_2.T - c) + offset_squared


def _get_log_marginal_likelihood_gp(amplitude_gaussian_squared: float,
                                    length_scale: float,
                                    noise_scale_squared: float,
                                    amplitude_linear_squared: float,
                                    offset_squared: float,
                                    c: float,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    distances_array: np.ndarray
                                    ) -> float:
    """
    Calculate the marginal log-likelihood of a Gaussian Process having a GaussianLinear Kernel

    :param amplitude_gaussian_squared: Parameter of the gaussian kernel
    :param length_scale: Parameter of the gaussian kernel
    :param noise_scale_squared: variance of the noise in the observations
    :param amplitude_linear_squared: Parameter of the Linear kernel (see above)
    :param offset_squared: Parameter of the Linear kernel (see above)
    :param c: Parameter of the Linear kernel (see above)
    :param X: data points of shape (N, 1) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the evaluation associated to a data point.
    y_i = f(x_i) + noise
    :param distances_array:
    :return: The log-marginal likelihood of the Gaussian Process whose kernel is defined by the parameters above
    """
    K = get_cov_matrix_gaussian_linear(amplitude_gaussian_squared, length_scale, amplitude_linear_squared, offset_squared, c, X,
                                       X, distances_array)

    K_noise = K + noise_scale_squared * np.identity(K.shape[0])

    K_noise_chol = np.linalg.cholesky(K_noise + 1e-3 * np.identity(K_noise.shape[0]))

    K_chol_inv = jax.scipy.linalg.solve_triangular(K_noise_chol, np.identity(K_noise_chol.shape[0]), lower=True)

    result = (
            0.5 * np.linalg.norm(np.dot(K_chol_inv, y)) ** 2
            + 0.5 * np.linalg.slogdet(K_noise)[1]
            + 0.5 * K.shape[0] * np.log(2 * np.pi)
    )

    return -1 * result


def expected_log_marginal_likelihood(mu: np.ndarray,
                                     A: np.ndarray,
                                     epsilon: np.ndarray,
                                     X: np.ndarray,
                                     y: np.ndarray
                                     ) -> float:
    """
    :param mu: mean of the posterior distribution approximated by variational inference
    :param A: Choleski matrix such that Sigma = A * A.T,
    where Sigma is the coveriance of the posterior distribution approximated by variational inference.
    :param epsilon: The samples from N(0, I) that will be used to generate samples from
    the approximated posterior N(mu, Sigma) by using the matrix A and the vector mu
    :param X: data points of shape (N, 1) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the evaluation associated to a data point.
    y_i = f(x_i) + noise
    :return: The expected log-likelihood. That expectation is calculated according to the approximated posterior
    N(mu, Sigma) by using the samples in epsilon.
    """
    S = epsilon.shape[0]
    N = y.shape[0]
    
    distance_array = get_distances_array(X, X)
    
    log_p_y_sum = 0
    
    for s in range(S):
        theta_s = mu + (A @ epsilon[s]).reshape((-1,1)) # (6,1)
        
        amplitude_gaussian_squared = onp.exp(theta_s[0]) ** 2
        length_scale = onp.exp(theta_s[1]) 
        noise_scale_squared = onp.exp(theta_s[2]) ** 2
        amplitude_linear_squared = onp.exp(theta_s[3]) ** 2
        offset_squared = onp.exp(theta_s[4]) ** 2
        c = theta_s[5]
    
        log_p_y = _get_log_marginal_likelihood_gp(amplitude_gaussian_squared, length_scale, noise_scale_squared, amplitude_linear_squared, offset_squared, c, X, y, distance_array)
        log_p_y_sum += log_p_y
    
    exp_log_lkd = log_p_y_sum/S
    
    return exp_log_lkd


def kl_div(mu: np.ndarray,
           A_chol: np.ndarray,
           sigma_prior: float
           ) -> float:
    """
    Computes the KL divergence between
    - the approximated posterior distribution N(mu, Sigma)
    - and the prior distribution on the parameters N(0, (sigma_prior ** 2) I)

    Instead of working directly with the covariance matrix Sigma, we will only deal with its Cholesky matrix A:
    It is the lower triangular matrix such that Sigma = A * A.T

    :param mu: mean of the posterior distribution approximated by variational inference
    :param A: Choleski matrix such that Sigma = A * A.T,
    where Sigma is the coveriance of the posterior distribution approximated by variational inference.
    :param sigma_prior: standard deviation of the prior on the parameters. We put the following prior on the parameters:
    N(mean=0, variance=(sigma_prior**2) I)
    :return: the value of the KL divergence
    """
    d = mu.shape[0]
    Sigma = A_chol @ A_chol.T
    det1 = onp.linalg.det(Sigma)
    det2 = onp.linalg.det((sigma_prior ** 2) * onp.eye(d))
    sigma_2_inv = (1/(sigma_prior ** 2)) * onp.eye(d)
    
    kl_div = 0.5 * (onp.log(det2/det1) - d + onp.trace(sigma_2_inv @ Sigma) + mu.T @ sigma_2_inv @ mu)

    return kl_div


def variational_inference_gp(X: np.ndarray,
                             y: np.ndarray,
                             num_samples_per_turn: int,
                             sigma_prior: float,
                             number_iterations: int,
                             ):
    """
    This function performs a variational inference procedure.

    Here we consider that our parameters follow a normal distribution N(mu, Sigma).
    Instead of working directly with the covariance matrix, we will only deal with its Cholesky matrix A:
    It is the lower triangular matrix such that Sigma = A * A.T

    At the end of each step, it yields the following elements (in this order):
    - the new estimated mu
    - the new estimated Sigma
    - the new estimated lower triangular matrix A (verifying A * A.T = Sigma)
    - mu_grad: the gradient of the marginal likelihood bound with respect to mu
    - A_grad: the gradient of the marginal likelihood bound with respect to A
    - epsilon: The samples from N(0, I) that were used to generate the samples from N(mu, Sigma)

    :param X: data points of shape (N, 1) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the evaluation associated to a data point.
    y_i = f(x_i) + noise
    :param num_samples_per_turn: number of samples of parameters to use at each step of the Blackbox Variational
    Inference Algorithm
    :param sigma_prior: standard deviation of the prior on the parameters. We put the following prior on the parameters:
    N(mean=0, variance=(sigma_prior**2) I)
    :param number_iterations: number of Blackbox Variational Inference steps before stopping
    """

    P = 6

    counter = 0
    mu = np.zeros(shape=(1, P)) + 0.01
    A = 0.1 * onp.identity(P)
    A[5, 5] = 0.01
    A = np.array(A)

    # Matrix used to make sure that the elements on the diagonal of A remain superior to 1e-5 at every step
    T = onp.full_like(A, -float('inf'))
    for i in range(P):
        T[i, i] = 1e-5

    epsilon = None
    mu_grad = None
    A_grad = None

    while counter < number_iterations:
        A_old = A
        mu_old = mu

        #############################
        # TODO : Complete Here for computing epsilon, mu_grad and A_grad
        #############################

        # Performing a gradient descent step on A and mu
        # (we make sure that the elements on the diagonal of A remain superior to 1e-5)
        A = np.maximum(A + (1. / (10 * counter + 500.)) * np.tril(A_grad), T)
        mu = mu + (1. / (10 * counter + 500.)) * mu_grad

        counter += 1
        if counter % 1 == 0:
            # Printing the highest change in parameters at that iteration
            print(f"counter: {counter} - {onp.max((onp.linalg.norm(mu_old - mu), onp.linalg.norm(A_old - A)))}\r")

        yield mu, A.dot(A.T), A, mu_grad, A_grad, epsilon


if __name__ == '__main__':
    onp.random.seed(207)
    obj = LinearSin(0.5)

    X = obj.get_uniform_dataset(21).reshape((-1, 1))
    y = obj(X)

    mu, Sigma = None, None

    for mu, Sigma, *_ in variational_inference_gp(X, y, num_samples_per_turn=10, sigma_prior=1., number_iterations=100):
        pass

    plot_vi_gp(obj, mu, Sigma, X, y)
