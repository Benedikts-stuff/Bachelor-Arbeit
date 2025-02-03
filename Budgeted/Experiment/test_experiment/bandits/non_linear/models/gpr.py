from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize as opt


class MyGPR(GaussianProcessRegressor):
    def __init__(self, kernel, alpha, normalize_y, n_restarts_optimizer,  _max_iter=100):
        super().__init__(kernel, alpha=alpha, normalize_y= normalize_y, n_restarts_optimizer= n_restarts_optimizer)
        self._max_iter = _max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            result = opt.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": self._max_iter}
            )
            return result.x, result.fun

        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)