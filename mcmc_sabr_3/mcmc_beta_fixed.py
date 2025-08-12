import numpy as np
from numba import njit
from mcmc_sabr_3.mcmc_vectorized import MCMCVectorized
import mcmc_sabr.config as config

@njit
def _transform_3_params(param_value):
    return np.array([
        np.log(param_value[0]),
        np.arctanh(param_value[1]),
        np.log(param_value[2])
    ])

@njit
def _inverse_transform_3_params(transformed_value):
    return np.array([
        np.exp(transformed_value[0]),
        np.tanh(transformed_value[1]),
        np.exp(transformed_value[2])
    ])

@njit
def _jacobian_det_3_params(param_value):
    alpha, rho, volvol = param_value
    return 1 / (alpha * (1 - rho**2) * volvol)

class MCMCBetaFixed(MCMCVectorized):
    """Implémentation du MCMC avec beta fixé."""

    def _initialize_sampler(self):
        self.param_names = ["alpha", "rho", "volvol"]
        self.n_params = 3
        self.params_bounds = config.THREE_PARAM_BOUNDS
        self.s_d = config.S_D_3_PARAMS
        print(f"Initialized Vectorized MCMC for 3 parameters (beta fixed at {config.FIXED_BETA}).")

    def transform_parameter(self, param_value):
        return _transform_3_params(param_value)

    def inverse_transform_parameter(self, transformed_value):
        return _inverse_transform_3_params(transformed_value)

    def compute_jacobian_det(self, param_value):
        return _jacobian_det_3_params(param_value)

    def get_one_param_jacobian_func(self, param_idx):
        # Ordre: alpha, rho, volvol
        funcs = [
            lambda x: 1 / x,
            lambda x: 1 / (1 - x**2),
            lambda x: 1 / x,
        ]
        return funcs[param_idx]
        
    def transform_one_parameter(self, value, param_index):
        transforms = [
            lambda x: np.log(x),
            lambda x: np.arctanh(x),
            lambda x: np.log(x)
        ]
        return transforms[param_index](value)

    def inverse_transform_one_parameter(self, value, param_index):
        transforms = [
            lambda x: np.exp(x),
            lambda x: np.tanh(x),
            lambda x: np.exp(x)
        ]
        return transforms[param_index](value)
    
    @njit
    def _inverse_transform_3_params(self,transformed_value):
        return np.array([
            np.exp(transformed_value[0]),
            np.tanh(transformed_value[1]),
            np.exp(transformed_value[2])
        ])