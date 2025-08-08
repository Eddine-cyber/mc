import numpy as np
from scipy import stats
from scipy.linalg import inv
from SABR.prapare_data import prepare_calibration_inputs
from SABR.Calibrate import calibrate_sabr, calibrate_sabr_with_beta, calibrate_sabr_all_maturities, calibrate_sabr_all_maturities_without_beta
from SABR.utils import compute_vol_sabr_derivative_analytical_all_maturities
import mcmc_sabr.config as config
from numba import njit


def load_market_data():
    """
    Load and prepare market volatility data for calibration.
    
    Returns:
        tuple: (maturity, forward, market_vols, strikes)
    """
    strikes, maturity_data = prepare_calibration_inputs(config.DATA_FILE_PATH, config.INITIAL_DATE)
    
    current_maturity = maturity_data[config.MATURITY_INDEX, 0]
    current_forward = maturity_data[config.MATURITY_INDEX, 1]
    current_market_vols = maturity_data[config.MATURITY_INDEX, 2:]
    
    return current_maturity, current_forward, current_market_vols, strikes

def get_initial_params_from_calibration(market_data, n_params=4, n_chains=4):
    """
    Get initial parameter values from calibration to market data.
    
    Returns:
        np.ndarray: Initial SABR parameters [alpha, beta, rho, volvol]
    """
    current_maturity, current_forward, current_market_vols, strikes = market_data
    if n_params == 4:
        theta_with_beta = calibrate_sabr_with_beta(current_maturity, strikes, current_forward, 
                                        current_market_vols, True, True)
        initial_params = np.zeros((n_chains, 4))
        # Créer des points de départ légèrement différents
        for i in range(n_chains):
            perturbation = 1 + (np.random.rand(4) - 0.5) * 0.2 * (i>0) # +/- 10%
            initial_params[i] = [theta_with_beta[0], theta_with_beta[1], theta_with_beta[2], theta_with_beta[3]] * perturbation
    else: # n_params == 3
        theta_without_beta = calibrate_sabr(current_maturity, strikes, current_forward, 
                                          current_market_vols, True, config.FIXED_BETA, True)
        initial_params = np.zeros((n_chains, 3))
         # Créer des points de départ légèrement différents
        for i in range(n_chains):
            perturbation = 1 + (np.random.rand(3) - 0.5) * 0.2 * (i>0) # +/- 10%
            initial_params[i] = np.array([theta_without_beta[0], theta_without_beta[1], theta_without_beta[2]]) * perturbation

    return initial_params

def get_initial_params_from_calibration_all_maturities(market_data_all_maturities, n_params=4, n_chains=4):
    """
    Get initial parameter values from calibration to market data.
    
    Returns:
        np.ndarray: Initial SABR parameters [alpha, beta, rho, volvol]
    """
    maturities, FwdImp, market_vols, strikes = market_data_all_maturities

    if n_params == 4:
        theta_with_beta = calibrate_sabr_all_maturities(strikes, market_vols, FwdImp, maturities) 
        initial_params = np.zeros((n_chains, 4))
        # Créer des points de départ légèrement différents
        for i in range(n_chains):
            perturbation = 1 + (np.random.rand(4) - 0.5) * 0.2 * (i>0) # +/- 10%
            initial_params[i] = [theta_with_beta[0], theta_with_beta[1], theta_with_beta[2], theta_with_beta[3]] * perturbation
    else: # n_params == 3
        theta_without_beta = calibrate_sabr_all_maturities_without_beta(strikes, market_vols, FwdImp, maturities, config.FIXED_BETA) 
        initial_params = np.zeros((n_chains, 3))
         # Créer des points de départ légèrement différents
        for i in range(n_chains):
            perturbation = 1 + (np.random.rand(3) - 0.5) * 0.2 * (i>0) # +/- 10%
            initial_params[i] = np.array([theta_without_beta[0], theta_without_beta[1], theta_without_beta[2]]) * perturbation

    return initial_params


@njit
def compute_jacobian(param_value):
    """
    Compute Jacobian matrix for parameter transformation.

                                [1/α    0          0          0    ]
    J = ∂φ/∂θ = [∂φᵢ/∂θⱼ]ᵢ,ⱼ =  [0      1/(β(1-β)) 0          0    ]
                                [0      0          1/1(-ρ²)   0    ]
                                [0      0          0          1/ν  ]

    
    Args:
        param_value (np.ndarray): [α, β, ρ, ν]
        
    Returns:
        float: Jacobian matrix
    """
    alpha, beta, rho, volvol = param_value
    J = np.array([
        [1 / alpha,                   0,                   0,           0],
        [0,           1 / (beta * (1 - beta)),             0,           0],
        [0,                           0,       1 / (1 - rho**2),        0],
        [0,                           0,                   0,      1 / volvol]
    ])

    return J
    
def compute_prior_mean_std(confidence_interval, param_bounds, transform_func):
    """Calcule la moyenne et l'écart-type du prior dans l'espace transformé."""
    transformed_bounds = np.array([
        transform_func(param_bounds.T[0]),
        transform_func(param_bounds.T[1])
    ]).T

    z = stats.norm.ppf(0.5 + (confidence_interval / 100) / 2)
    mean_array = np.mean(transformed_bounds, axis=1)
    std_array = (transformed_bounds[:, 1] - transformed_bounds[:, 0]) / (2 * z)
    return mean_array, std_array

def compute_fisher_information_matrix(param, strikes, forward, t_maturity):
    """
    Calcule la matrice d'information de Fisher pour SABR
    I(θ)ᵢⱼ = (1/σ²) Σₖ [∂SABR(Kₖ)/∂θᵢ × ∂SABR(Kₖ)/∂θⱼ]
    """
    n_params = len(param)
    fisher_matrix = np.zeros((n_params, n_params))
    # Calcul des gradients pour chaque paramètre
    gradients = []
    for i in range(n_params):
        grad_i = compute_vol_sabr_derivative_analytical_all_maturities(i, param, strikes, forward, t_maturity)
        gradients.append(grad_i)
    for i in range(n_params):
        for j in range(n_params):
            # Somme sur tous les strikes
            fisher_matrix[i, j] = np.sum(gradients[i] * gradients[j]) / (config.OBSERVATION_ERROR**2)

    return fisher_matrix

def compute_covariance_matrix_from_hessienne(param, strikes, forward, t_maturity, regularization=1e-8):
    
    fisher_matrix = compute_fisher_information_matrix(param, strikes, forward, t_maturity)
    
    jacobian_matrix = compute_jacobian(param)
    
    fisher_transformed = jacobian_matrix.T @ fisher_matrix @ jacobian_matrix # I_φ = J^T I_θ J
    
    fisher_regularized = fisher_transformed + regularization * np.eye(len(param)) # matrice soit définie positive
    
    cov_asymptotic = inv(fisher_regularized)
    
    cov_matrix = 0.1 * cov_asymptotic
    
    return cov_matrix


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

