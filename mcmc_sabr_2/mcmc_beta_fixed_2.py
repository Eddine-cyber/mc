import numpy as np
from mcmc_sabr_2.mcmc_vectorized_2 import MCMCVectorized
import mcmc_sabr.config as config
from mcmc_sabr.utils import _transform_3_params, _inverse_transform_3_params, _jacobian_det_3_params

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

    def inverse_transform_one_parameter(self, value, param_index):
        transforms = [
            lambda x: np.exp(x),
            lambda x: np.tanh(x),
            lambda x: np.exp(x)
        ]
        return transforms[param_index](value)
        
    def transform_one_parameter(self, value, param_index):
        transforms = [
            lambda x: np.log(x),
            lambda x: np.arctanh(x),
            lambda x: np.log(x)
        ]
        return transforms[param_index](value)