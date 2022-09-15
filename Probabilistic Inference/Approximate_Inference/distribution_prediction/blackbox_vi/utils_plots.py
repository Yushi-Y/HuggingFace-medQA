import numpy as onp
from jax import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from scipy.stats import multivariate_normal

from gaussian_process import GaussianProcess
from kernels.gaussian_linear_kernel import GaussianLinearKernel


def probability_class_1(x, theta):
    from distribution_prediction.blackbox_vi.blackbox_vi_logistics import sigmoid

    return sigmoid(x, theta)


def plot_vi_logistics(number_points_per_class, interactive=False, interval_plot=1, number_iterations=1000):
    from distribution_prediction.blackbox_vi.blackbox_vi_logistics import variational_inference_logistics

    def _plot(fig, ax1, ax2, mean, sigma, array_samples_theta, interactive=False):
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
        predictions = onp.mean(probability_class_1(Z, array_samples_theta), axis=1)
        predictions = predictions.reshape(100, 100)
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

        x_prior = np.linspace(-3, 3, 100)
        y_prior = np.linspace(-3, 3, 100)
        X_prior, Y_prior = np.meshgrid(x_prior, y_prior)
        Z = np.dstack((X_prior, Y_prior))
        Z = Z.reshape(-1, 2)
        prior_values = multivariate_normal.pdf(Z, np.zeros(2), np.identity(2))
        prior_values = prior_values.reshape(100, 100)

        std_x = onp.sqrt(sigma[0, 0])
        std_y = onp.sqrt(sigma[1, 1])
        x_posterior = np.linspace(mean[0] - 3 * std_x, mean[0] + 3 * std_x, 100)
        y_posterior = np.linspace(mean[1] - 3 * std_y, mean[1] + 3 * std_y, 100)
        X_post, Y_post = np.meshgrid(x_posterior, y_posterior)

        Z_post = np.dstack((X_post, Y_post)).reshape(-1, 2)
        posterior_values = multivariate_normal.pdf(Z_post, mean, sigma)
        posterior_values = posterior_values.reshape(100, 100)

        ax2.contour(X_post, Y_post, posterior_values)
        ax2.contour(X_, Y_, prior_values, cmap="inferno")
        ax2.set_title("Two contour plots respectively showing\n"
                      "The prior and the approximated posterior distributions")

        plt.pause(0.001)
        if interactive:
            if np.size(predictions):
                colorbar.remove()

        return True

    mean_1 = onp.array([-2, 2])
    mean_2 = onp.array([2, -2])

    X_1 = onp.random.randn(number_points_per_class, 2) + mean_1
    X_2 = onp.random.randn(number_points_per_class, 2) + mean_2

    X = onp.vstack((X_1, X_2))
    y_1 = onp.ones(shape=(X_1.shape[0], 1))
    y_2 = onp.zeros(shape=(X_2.shape[0], 1))
    y = onp.vstack((y_1, y_2))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    #######################
    mean, sigma = None, None
    count = 0
    for mean, sigma, *_ in variational_inference_logistics(X, y, 1000, sigma_prior=1.,
                                                           number_iterations=number_iterations):
        mean = mean.flatten()
        array_samples_theta = multivariate_normal.rvs(mean, sigma, 1000)
        if interactive and count % interval_plot == 0:
            _plot(fig, ax1, ax2, mean, sigma, array_samples_theta, interactive=interactive)
        count += 1
    else:
        array_samples_theta = multivariate_normal.rvs(mean, sigma, 1000)
        _plot(fig, ax1, ax2, mean, sigma, array_samples_theta, interactive=False)
    ########################

    xlim = (-4., 4.)
    ylim = (-4., 4.)
    xlist = onp.linspace(*xlim, 100)
    ylist = onp.linspace(*ylim, 100)
    X_, Y_ = onp.meshgrid(xlist, ylist)
    Z = onp.dstack((X_, Y_))
    Z = Z.reshape(-1, 2)
    predictions = onp.mean(probability_class_1(Z, array_samples_theta),
                           axis=1)
    predictions = predictions.reshape(100, 100)
    print("finished")

    fig, ax = plt.subplots(1, 1)
    CS = plt.contourf(X_, Y_, predictions, levels=20)
    colorbar = plt.colorbar(CS)
    plt.scatter(X_1[:, 0], X_1[:, 1])
    plt.scatter(X_2[:, 0], X_2[:, 1])
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.show()


def plot_vi_gp(obj, mu, Sigma, X, y):
    gp = GaussianProcess(GaussianLinearKernel(0., 0., 0., 0., 0., 0.), X, y)

    xlim, = obj.boundaries
    x_gt = onp.linspace(xlim[0], xlim[1], 100)
    xx = onp.linspace(xlim[0] - 2, xlim[1] + 2, 200)

    plt.plot(x_gt, obj.evaluate_without_noise(x_gt), c='c')

    plt.title(f"Gaussian Process Regression")
    mu = mu.flatten()
    for _ in range(500):
        sample_gp_parameter = onp.random.multivariate_normal(mu, Sigma)
        gp.set_kernel_parameters(*sample_gp_parameter.flatten())
        function_sample = gp.get_sample(xx.reshape((-1, 1)))
        plt.plot(xx, function_sample, alpha=0.3, c='C0')
        plt.scatter(gp.array_dataset,
                    gp.array_objective_function_values,
                    c='m',
                    marker="+",
                    zorder=1000,
                    s=(30,))
        plt.pause(0.01)
    plt.show()