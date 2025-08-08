import numpy as np
from mcmc_sabr.mcmc_vectorized import MCMCVectorized

class MCMCSequential(MCMCVectorized):
    """Implémentation du MCMC séquentiel (Metropolis-within-Gibbs)."""

    def _initialize_sampler(self):
        super()._initialize_sampler()
        # Le taux d'acceptance est suivi par paramètre, pas seulement par chaîne
        self.acceptance_counts = np.zeros((self.n_chains, self.n_params))
        print("Initialized Sequential MCMC.")

    def propose_one_parameter(self, current_params, param_idx, chain_idx):
        """Propose un nouveau jeu de paramètres en modifiant un seul."""
        proposed_params = current_params.copy()
        
        # Transformation, proposition, et transformation inverse pour un seul paramètre
        current_val = current_params[param_idx]
        transformed_val = self.transform_one_parameter(current_val, param_idx)
        
        # Utiliser l'écart-type de la diagonale de la matrice de covariance de proposition
        proposal_std = np.sqrt(self.cov_proposal[chain_idx, param_idx, param_idx])
        proposed_transformed = np.random.normal(transformed_val, proposal_std)
        
        proposed_params[param_idx] = self.inverse_transform_parameter(proposed_transformed, param_idx)
        return proposed_params

    def compute_one_param_proposal_ratio(self, current_val, proposed_val, param_idx):
        """Calcule le ratio de proposition pour une mise à jour uni-paramètre."""
        ratios = {
            0: lambda c, p: p / c,  # alpha
            1: lambda c, p: (p * (1 - p)) / (c * (1 - c)),  # beta
            2: lambda c, p: (1 - p**2) / (1 - c**2),  # rho
            3: lambda c, p: p / c,  # volvol
        }
        return ratios[param_idx](current_val, proposed_val)

    def compute_one_param_acceptance_prob(self, current_params, proposed_params, param_idx, chain_idx):
        """Calcule la probabilité d'acceptance pour une mise à jour uni-paramètre."""
        if not self.check_parameter_bounds(proposed_params):
            return 0.0
        
        log_lik_ratio = self.compute_log_likelihood(proposed_params) - self.compute_log_likelihood(current_params)
        
        prior_ratio = self.compute_prior_density(proposed_params, chain_idx) / self.compute_prior_density(current_params, chain_idx)

        proposal_ratio = self.compute_one_param_proposal_ratio(
            current_params[param_idx], proposed_params[param_idx], param_idx
        )
        
        return min(1.0, np.exp(log_lik_ratio) * prior_ratio * proposal_ratio)

    def _run_one_iteration(self, iteration):

        param_indices = np.arange(self.n_params)
        np.random.shuffle(param_indices) # Ordre de mise à jour aléatoire

        for j in range(self.n_chains):
            for param_idx in param_indices:
                proposed = self.propose_one_parameter(self.current_params[j], param_idx, j)
                alpha = self.compute_one_param_acceptance_prob(
                    self.current_params[j], proposed, param_idx, j
                )

                if np.random.uniform() < alpha:
                    self.current_params[j] = proposed
                    self.acceptance_counts[j, param_idx] += 1