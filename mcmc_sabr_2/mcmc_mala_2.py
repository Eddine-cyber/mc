import numpy as np
from scipy import stats
from mcmc_sabr_2.mcmc_vectorized_2 import MCMCVectorized
import mcmc_sabr.config as config
from SABR.sabr import vol_sabr_hagan
from SABR.utils import compute_vol_sabr_grad_analytical


class MCMCMala(MCMCVectorized):
    """Implémentation du MCMC avec gradient (MALA) en utilisant le gradient analytique."""

    def _initialize_sampler(self):
        super()._initialize_sampler()
        self.langevin_error = config.LANGEVIN_ERROR
        print("Initialized MALA MCMC with analytical gradient.")


    def compute_log_likelihood_gradient_analytical(self, param_transformed):
        """
        Calcule le gradient de la log-vraisemblance par rapport aux paramètres transformés 'φ'
        en utilisant le gradient analytique de SABR par rapport aux paramètres 'θ'.
        
        ∇φ log L = (∂θ/∂φ)ᵀ * ∇θ log L
        """
        params_theta = self.inverse_transform_parameter(param_transformed)

        maturities, FwdImp, market_vols, strikes = self.market_data_all_maturities
        
        n_maturities = len(maturities)
        n_strikes = len(strikes)
        
        all_residuals = np.zeros((n_maturities, n_strikes))
        all_grad_sigma = np.zeros((n_maturities, n_strikes, self.n_params))
        
        for i in range(n_maturities):
            t_maturity, forward = maturities[i], FwdImp[i]
            if self.n_params == 4:
                sabr_vols = vol_sabr_hagan(strikes, forward, t_maturity, *params_theta)
            else :
                alpha, rho, volvol = params_theta
                sabr_vols = vol_sabr_hagan(strikes, forward, t_maturity, alpha, config.FIXED_BETA, rho, volvol)
            
            all_residuals[i] = sabr_vols - market_vols[i]
            # Calculer le gradient de la vol SABR par rapport à θ
            all_grad_sigma[i] = compute_vol_sabr_grad_analytical(params_theta, strikes, forward, t_maturity)
        
        residuals_expanded = all_residuals[:, :, np.newaxis]  # Shape: (n_maturities, n_strikes, 1)

        # Calculer le gradient de la log-vraisemblance par rapport à θ
        # ∂L/∂θ_i = (1/ε²) * Σ_k (residual_k * ∂σ_sabr,k/∂θ_i)   
        grad_contributions = residuals_expanded * all_grad_sigma  # Shape: (n_maturities, n_strikes, n_params)
        grad_L_wrt_theta = np.sum(grad_contributions, axis=(0, 1)) / (config.OBSERVATION_ERROR**2)
        
        # Appliquer la règle de la chaîne pour obtenir le gradient par rapport à φ
        # ∇φ L = ∇θ L * (∂θ/∂φ) car la Jacobienne est diagonale
        if self.n_params == 4:
            alpha, beta,  rho, volvol = params_theta
            d_theta_d_phi = np.array([
                alpha,              # ∂α/∂log(α)
                beta * (1 - beta),  # ∂β/∂logit(β)
                1 - rho**2,         # ∂ρ/∂arctanh(ρ)
                volvol              # ∂ν/∂log(ν)
            ])
        else :
            alpha, rho, volvol = params_theta
            d_theta_d_phi = np.array([
                alpha,              # ∂α/∂log(α)
                1 - rho**2,         # ∂ρ/∂arctanh(ρ)
                volvol              # ∂ν/∂log(ν)
            ])
        grad_L_wrt_phi = grad_L_wrt_theta * d_theta_d_phi
        
        return grad_L_wrt_phi

    def propose_new_parameter(self, current_param, chain_idx):
        """Propose de nouveaux paramètres en utilisant la dérive de Langevin."""
        current_transformed = self.transform_parameter(current_param)
        
        grad_ll = self.compute_log_likelihood_gradient_analytical(current_transformed)
        
        # Matrice de covariance de la proposition)
        G_inv = self.cov_proposal[chain_idx]
        
        # mu = φ_t + 0.5 * ε * G⁻¹ * ∇φ log L(φ_t)
        mu = current_transformed + 0.5*self.langevin_error * G_inv @ grad_ll
        
        # Proposition avec covariance  ε * G⁻¹
        proposed_transformed = np.random.multivariate_normal(mu, self.langevin_error * G_inv)
        
        return self.inverse_transform_parameter(proposed_transformed)

    def compute_proposition_kernel_log_pdf(self, from_transformed, to_transformed, chain_idx):
        """Calcule le log de la densité de la proposition q(to | from)."""
        grad_ll = self.compute_log_likelihood_gradient_analytical(from_transformed)
        G_inv = self.cov_proposal[chain_idx]
        mu = from_transformed + 0.5*self.langevin_error * G_inv @ grad_ll
        cov = self.langevin_error * G_inv
        
        return stats.multivariate_normal.logpdf(to_transformed, mean=mu, cov=cov)

    def compute_proposal_ratio(self, current_param, proposed_param, chain_idx):
        """Calcule le ratio de proposition pour MALA."""
        # Correction Jacobienne standard pour le changement de variable
        jacobian_ratio = super().compute_proposal_ratio(current_param, proposed_param)

        # Correction pour l'asymétrie de la proposition MALA
        current_transformed = self.transform_parameter(current_param)
        proposed_transformed = self.transform_parameter(proposed_param)

        log_q_reverse = self.compute_proposition_kernel_log_pdf(proposed_transformed, current_transformed, chain_idx)
        log_q_forward = self.compute_proposition_kernel_log_pdf(current_transformed, proposed_transformed, chain_idx)
        
        mala_ratio = np.exp(log_q_reverse - log_q_forward)

        return jacobian_ratio * mala_ratio
    
    def compute_acceptance_probability(self, current_param, proposed_param, chain_idx):
        """Calcule la probabilité d'acceptance de Metropolis-Hastings pour MALA."""
        if not self.check_parameter_bounds(proposed_param):
            return 0.0
        
        current_prior = self.compute_prior_density(current_param, chain_idx)
        proposed_prior = self.compute_prior_density(proposed_param, chain_idx) 
        log_lik_ratio = self.compute_log_likelihood(proposed_param) - self.compute_log_likelihood(current_param)
        prior_ratio = proposed_prior / current_prior
        proposal_ratio = self.compute_proposal_ratio(current_param, proposed_param, chain_idx)
        
        return min(1.0, np.exp(log_lik_ratio) * prior_ratio * proposal_ratio)