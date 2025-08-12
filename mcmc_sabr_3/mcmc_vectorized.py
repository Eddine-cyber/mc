import numpy as np
from numba import njit
from mcmc_sabr_3.mcmc_base import MCMCBase
import mcmc_sabr.config as config

@njit
def _transform_4_params(param_value):
    return np.array([
        np.log(param_value[0]),
        np.log(param_value[1] / (1 - param_value[1])),
        np.arctanh(param_value[2]),
        np.log(param_value[3])
    ])

@njit
def _inverse_transform_4_params(transformed_value):
    return np.array([
        np.exp(transformed_value[0]),
        1 / (1 + np.exp(-transformed_value[1])),
        np.tanh(transformed_value[2]),
        np.exp(transformed_value[3])
    ])

@njit
def _jacobian_det_4_params(param_value):
    alpha, beta, rho, volvol = param_value
    return 1 / (alpha * beta * (1 - beta) * (1 - rho**2) * volvol)


class MCMCVectorized(MCMCBase):
    """Implémentation du MCMC vectoriel (Metropolis-Hastings standard)."""

    def _initialize_sampler(self):
        self.param_names = ["alpha", "beta", "rho", "volvol"]
        self.n_params = 4
        self.params_bounds = config.FOUR_PARAM_BOUNDS
        self.s_d = config.S_D_4_PARAMS
        print("Initialized Vectorized MCMC for 4 parameters.")

    def transform_parameter(self, param_value):
        return _transform_4_params(param_value)

    def inverse_transform_parameter(self, transformed_value):
        return _inverse_transform_4_params(transformed_value)

    def compute_jacobian_det(self, param_value):
        return _jacobian_det_4_params(param_value)

    def get_one_param_jacobian_func(self, param_idx):

        # Ordre: alpha, beta, rho, volvol
        funcs = [
            lambda x: 1 / x,
            lambda x: 1 / (x * (1 - x)),
            lambda x: 1 / (1 - x**2),
            lambda x: 1 / x,
        ]
        return funcs[param_idx]
        
    def transform_one_parameter(self, value, param_index):
        transforms = [
            lambda x: np.log(x),
            lambda x: np.log(x / (1 - x)),
            lambda x: np.arctanh(x),
            lambda x: np.log(x)
        ]
        return transforms[param_index](value)
    
    def inverse_transform_one_parameter(self, value, param_index):
        transforms = [
            lambda x: np.exp(x),
            lambda x: 1/(1 + np.exp(-x)),
            lambda x: np.tanh(x),
            lambda x: np.exp(x)
        ]
        return transforms[param_index](value)

    def propose_new_parameter(self, current_param, chain_idx):
        """Propose un nouveau jeu de paramètres."""
        current_transformed = self.transform_parameter(current_param)
        proposed_transformed = np.random.multivariate_normal(current_transformed, self.cov_proposal[chain_idx])
        return self.inverse_transform_parameter(proposed_transformed)

    def compute_proposal_ratio(self, current_param, proposed_param):
        """Ratio de proposition avec correction Jacobienne."""
        return self.compute_jacobian_det(current_param) / self.compute_jacobian_det(proposed_param)

    def compute_acceptance_probability(self, current_param, proposed_param, chain_idx):
        """Calcule la probabilité d'acceptance de Metropolis-Hastings."""
        if not self.check_parameter_bounds(proposed_param):
            return 0.0
        
        log_lik_ratio = self.compute_log_likelihood(proposed_param) - self.compute_log_likelihood(current_param)
        
        prior_ratio = self.compute_prior_density(proposed_param, chain_idx) / self.compute_prior_density(current_param, chain_idx)
        
        proposal_ratio = self.compute_proposal_ratio(current_param, proposed_param)
        
        return min(1.0, np.exp(log_lik_ratio) * prior_ratio * proposal_ratio)

    def _run_one_iteration(self, iteration):
        # Mise à jour adaptative de la covariance
        if self.adapt_cov and iteration > 0 and iteration % self.n0 == 0:
            k = iteration // self.n0
            for j in range(self.n_chains):
                self.cov_proposal[j], self.mean_proposal[j] = self.update_cov_matrix_proposal(
                    k, self.n0, self.cov_proposal[j], self.mean_proposal[j], self.new_samples_buffer[j]
                )

        # Proposition et acceptation/rejet pour chaque chaîne
        for j in range(self.n_chains):
            proposed = self.propose_new_parameter(self.current_params[j], j)
            acceptance_prob = self.compute_acceptance_probability(self.current_params[j], proposed, j)

            if np.random.uniform() < acceptance_prob:
                self.current_params[j] = proposed
                self.acceptance_counts[j] += 1
            
            # Stocker l'échantillon transformé pour la mise à jour de la covariance
            if self.adapt_cov:
                self.new_samples_buffer[j, iteration % self.n0] = self.transform_parameter(self.current_params[j])