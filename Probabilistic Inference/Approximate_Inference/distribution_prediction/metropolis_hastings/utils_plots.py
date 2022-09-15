import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from scipy.stats import multivariate_normal




def plot_metropolis_hastings_logistics(number_expected_iterations, interactive=False, sigma_exploration_mh=1, sigma_prior=1, number_points_per_class=5):
    from distribution_prediction.metropolis_hastings.metropolis_hastings_logistic import get_predictions, \
        metropolis_hastings
    def _plot(array_samples_data, fig, ax1, ax2, interactive=False):
        colorbar = None
        colorbar_2 = None

        plt.gca()
        # plt.cla()
        # plt.clf()
        fig.clear()
        fig.add_axes(ax1)
        fig.add_axes(ax2)

        plt.cla()

        xlim = (-5., 5.)
        ylim = (-5., 5.)
        xlist = np.linspace(*xlim, 100)
        ylist = np.linspace(*ylim, 100)
        X_, Y_ = np.meshgrid(xlist, ylist)
        Z = np.dstack((X_, Y_))
        Z = Z.reshape(-1, 2)
        predictions = get_predictions(Z, array_samples_theta)
        if np.size(predictions):
            predictions = predictions.reshape(100, 100)
        else:
            return False
        # print("finished")
        ax1.clear()
        if np.size(predictions):
            CS = ax1.contourf(X_, Y_, predictions, cmap="cividis")
        ax1.scatter(X_1[:, 0], X_1[:, 1])
        ax1.scatter(X_2[:, 0], X_2[:, 1])
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.set_title("Predicted probability of belonging to C_1")
        ax3 = fig.add_axes(Bbox([[0.43, 0.11], [0.453, 0.88]]))
        if np.size(predictions):
            colorbar = fig.colorbar(CS, cax=ax3, )
        ax1.set_position(Bbox([[0.125, 0.11], [0.39, 0.88]]))

        if np.size(array_samples_theta):
            colors = np.arange(1, array_samples_theta.shape[0] + 1)
            CS_2 = ax2.scatter(array_samples_theta[:, 0], array_samples_theta[:, 1], c=colors)
            colorbar_2 = plt.colorbar(CS_2, ax=ax2)
        x_prior = np.linspace(-3, 3, 100)
        y_prior = np.linspace(-3, 3, 100)
        X_prior, Y_prior = np.meshgrid(x_prior, y_prior)
        Z = np.dstack((X_prior, Y_prior))
        Z = Z.reshape(-1, 2)
        prior_values = multivariate_normal.pdf(Z, np.zeros(2), np.identity(2))
        prior_values = prior_values.reshape(100, 100)


        ax2.contour(X_, Y_, prior_values, cmap="inferno")
        ax2.set_title("Samples from the posterior distribution\n"
                      "The contour plot shows the prior distribution.")

        plt.pause(0.001)
        if interactive:
            if np.size(predictions):
                colorbar.remove()
                colorbar_2.remove()

        return True

    mean_1 = np.array([-2, 2])
    mean_2 = np.array([2, -2])

    X_1 = np.random.randn(number_points_per_class, 2) + mean_1
    X_2 = np.random.randn(number_points_per_class, 2) + mean_2

    X = np.vstack((X_1, X_2))

    y_1 = np.ones(shape=(X_1.shape[0], 1))
    y_2 = np.zeros(shape=(X_2.shape[0], 1))
    y = np.vstack((y_1, y_2))

    #######################
    # mean, sigma = variational_inference(X, y, 200)
    # array_samples_theta = multivariate_normal.rvs(mean, sigma, 1000)
    ########################
    fig, (ax1, ax2) = plt.subplots(1, 2)
    is_infinite_loop = True
    for index, (is_accepted, array_samples_theta, *_) in enumerate(metropolis_hastings(X, y.reshape(-1, 1), number_expected_iterations, sigma_exploration_mh=1, sigma_prior=1)):
        if interactive and is_accepted:
            _plot(array_samples_theta, fig, ax1, ax2, interactive)

        if is_accepted:
            is_infinite_loop = False

        if is_infinite_loop and index == 100000:
            print("No sample found before 100000th iteration, stopping...")
            return
    else:
        _plot(array_samples_theta, fig, ax1, ax2, interactive)
    plt.show()
