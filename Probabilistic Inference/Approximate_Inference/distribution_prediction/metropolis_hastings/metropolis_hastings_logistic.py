import numpy as np
from scipy.stats import bernoulli, multivariate_normal

from distribution_prediction.metropolis_hastings.utils_plots import plot_metropolis_hastings_logistics
from distribution_prediction.utils import sigmoid

def get_log_upper_proba_distribution(X: np.ndarray,
                                     y: np.ndarray,
                                     theta: np.ndarray,
                                     sigma_prior: float
                                     ) -> float:
    """
    This functions evaluates log( p_1(theta | X, y) ) where:
     - p_1 = Z * p
     - p is the posterior distribution
     - p_1 is easy to calculate

    You may use the sigmoid function in the file utils.py
    BE CAREFUL: be sure to reshape theta before passing it as an argument to the function sigma!!

    :param X: data points of shape (N, 2) where N is the number of data points, and 2 is the number of components for
    each data point x. Each row of X represents one data point.
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param theta: parameters at which we evaluate p_1. In our example, it is a numpy array (row vector) of shape (2,).
    :param sigma_prior: standard deviation of the prior on the parameters
    :return: log( p_1(theta | X, y) )
    """
    N = X.shape[0]
    theta_new = theta.reshape((1, -1))
    ber_prob = sigmoid(X, theta_new) # (N, 1)
    
    p_y = 1
    for n in range(N):
        p_yi = bernoulli.pmf(y[n][0], ber_prob[n][0])
        p_y *= p_yi
        
    p_theta = multivariate_normal.pdf(theta, [0, 0], sigma_prior**2)
    
    return np.log(p_y*p_theta)


def metropolis_hastings(X: np.ndarray,
                        y: np.ndarray,
                        number_expected_iterations: int,
                        sigma_exploration_mh: float = 1,
                        sigma_prior: float = 1):
    """
    Performs a Metropolis Hastings procedure.
    This function is a generator. After each step, it should yield a tuple containing the following elements
    (in this order):
    -  is_sample_accepted (type: bool) which indicates if the last sample from the proposal density has been accepted
    -  np.array(list_kept_thetas): numpy array of size (I, 2) where I represents the total number of previous
    iterations, and 2 is the number of components in theta in this logistic regression task.
    -  newly_sampled_theta: in this example, numpy array of size (2,)
    -  u (type: float): last random number used for deciding if the newly_sampled_theta should be accepted or not.


    :param X: data points of shape (N, 2) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param number_expected_iterations: Number of iterations for the Metropolis Hastings procedure
    :param sigma_exploration_mh: Standard deviation of the proposal density.
    We consider that the proposal density corresponds to a multivariate normal distribution, with:
    - mean = null vector
    - covariance matrix = (sigma_proposal_density ** 2) identity matrix
    :param sigma_prior: standard deviation of the prior on the parameters
    """

    # ----- These are some the variables you should manipulate in the main loop of that function ----------

    # Add one value of sampled theta to this list before the "yield".
    # (1) if newly_sampled_theta is accepted, you should add newly_sampled_theta to this list
    # (2) if newly_sampled_theta is rejected, you should add the last accepted sampled theta.
    list_kept_thetas = []

    # newly_sampled_theta = None  # Last sampled parameters (from the proposal density q)

    # is_sample_accepted = False  # Should be True if and only if the last sample has been accepted

    u = np.random.rand()  # Random number used for deciding if newly_sampled_theta should be accepted or not

    first_theta = np.zeros(X.shape[1])

    # -------------------------------------------------------------------------------------------------

    while len(list_kept_thetas) < number_expected_iterations:
        #########################
        if len(list_kept_thetas) == 0:
            theta_t = first_theta
        else:
            theta_t = np.array(list_kept_thetas[-1])
            
        newly_sampled_theta = multivariate_normal.rvs(theta_t, sigma_exploration_mh**2, 1)
                
        T_theta_t = multivariate_normal.pdf(theta_t, newly_sampled_theta, sigma_exploration_mh**2)
        T_theta = multivariate_normal.pdf(newly_sampled_theta, theta_t, sigma_exploration_mh**2)
        
        log_p_theta = get_log_upper_proba_distribution(X, y, newly_sampled_theta, sigma_prior)
        log_p_theta_t = get_log_upper_proba_distribution(X, y, theta_t, sigma_prior)

        u = np.random.rand()
        
        threshold = (T_theta_t * np.exp(log_p_theta))/(T_theta * np.exp(log_p_theta_t))       
        if threshold >= u:
            list_kept_thetas.append(newly_sampled_theta)
            is_sample_accepted = True
            
        else:
            is_sample_accepted = False
            list_kept_thetas.append(theta_t)
        #########################

        # print(np.array(list_kept_thetas).shape)

        yield is_sample_accepted, np.array(list_kept_thetas), newly_sampled_theta, u


def get_predictions(X_star: np.ndarray,
                    array_samples_theta: np.ndarray
                    ) -> np.ndarray:
    """
    :param X_star: array of data points of shape (N, 2) where N is the number of data points.
    There is one data point per row
    :param array_samples_theta: np array of shape (M, 2) where M is the number of sampled set of parameters
    generated by the Metropolis-Hastings procedure. Each row corresponds to one sampled theta.
    :return: estimated predictions at each point in X_star: p(C_1|X,y,x_star)=p(y_star=1|X,y,x_star),
    where each x_star corresponds to a row in X_star. The result should be a column vector of shape (N, 1), its i'th
    row should be equal to the prediction p(C_1|X,y,x_star_i) where x_star_i corresponds to the i'th row in X_star
    """
    N = X_star.shape[0]
    M = array_samples_theta.shape[0]

    sigmoid_matrix = sigmoid(X_star, array_samples_theta) # N*M
    p_y_pred = []
    
    for n in range(N):
        p_y = 0
        for m in range(M):
            p_y += sigmoid_matrix[n][m]
        ave_p_y = p_y/M
        p_y_pred.append(ave_p_y)
    
    return np.array([p_y_pred]) # Return a row vector to pass the LabTS test
    


if __name__ == '__main__':
    plot_metropolis_hastings_logistics(number_expected_iterations=200,
                                       interactive=True,
                                       sigma_exploration_mh=1,
                                       sigma_prior=1,
                                       number_points_per_class=25)
