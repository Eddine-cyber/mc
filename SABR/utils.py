import numpy as np
from numba import njit

def compute_vol_sabr_derivative_analytical(param_idx, param, strikes, forward, t_maturity):
    # for one maturity 
    alpha, beta, rho, nu = param
    deriv = np.zeros(len(strikes))

    mask = np.isclose(strikes, forward, atol=1e-6)
    exists = np.any(mask)

    if exists:
        # Trouver l'indice où K == f
        indice = np.where(mask)[0]
        for i, K in enumerate(strikes):        
            if i == indice:
                # Cas ATM
                deriv[i] = compute_atm_gradient(param_idx, alpha, beta, rho, nu, forward, t_maturity)
            else:
                # Cas général
                deriv[i] = compute_general_gradient(param_idx, alpha, beta, rho, nu, K, forward, t_maturity)
    else :
        for i, K in enumerate(strikes):        
            deriv[i] = compute_general_gradient(param_idx, alpha, beta, rho, nu, K, forward, t_maturity)

    return deriv

def compute_vol_sabr_derivative_analytical_all_maturities(param_idx, param, strikes, forwards, maturities):
    n_maturities = len(maturities)
    deriv = np.zeros((n_maturities, len(strikes)))
    for i in range(n_maturities):
        maturity = maturities[i]
        forward = forwards[i]
        deriv[i] = compute_vol_sabr_derivative_analytical(param_idx, param, strikes, forward, maturity)

    return deriv


def compute_vol_sabr_grad_analytical(param, strikes, forward, t_maturity):
    grad = np.zeros((len(strikes), len(param)))
    if len(param) == 4:
        for i in range (len(param)):
            grad[:, i] = compute_vol_sabr_derivative_analytical(i, param, strikes, forward, t_maturity) # c'est un vecteur de len: len(strikes)
        return grad
    elif len(param) == 3:
        for i in range (len(param)):
            alpha, rho, volvol = param
            import mcmc_sabr.config as config
            new_param = alpha, config.FIXED_BETA, rho, volvol # c'est fait pour ne pas redefinir la fonction ci dessous
            if i == 0:
                grad[:, i] = compute_vol_sabr_derivative_analytical(i, new_param, strikes, forward, t_maturity) # c'est un vecteur de len: len(strikes)
            else :
                grad[:, i] = compute_vol_sabr_derivative_analytical(i+1, new_param, strikes, forward, t_maturity) # c'est un vecteur de len: len(strikes) 
        return grad




@njit
def compute_atm_gradient(param_idx, alpha, beta, rho, nu, f, t):
    """
    Calcule le gradient ATM 
    """
    
    # Termes de base
    f_power_1_minus_beta = f**(1-beta)
    f_power_2_minus_2beta = f**(2-2*beta)
    
    # Terme principal
    main_term = alpha / f_power_1_minus_beta
    
    # Termes de correction
    term1 = (1-beta)**2 / 24 * alpha**2 / f_power_2_minus_2beta
    term2 = 1/4 * rho * beta * alpha * nu / f_power_1_minus_beta
    term3 = (2 - 3*rho**2) / 24 * nu**2
    
    correction = 1 + (term1 + term2 + term3) * t
    
    if param_idx == 0:  # Alpha
        grad_main = 1 / f_power_1_minus_beta
        grad_correction = ((1-beta)**2 / 12 * alpha / f_power_2_minus_2beta + 
                          1/4 * rho * beta * nu / f_power_1_minus_beta) * t
        return grad_main * correction + main_term * grad_correction
    
    elif param_idx == 1:  # Beta
        log_f = np.log(f)
        
        # ∂/∂β [α/f^(1-β)]
        grad_main = alpha * log_f / f_power_1_minus_beta
        
        # ∂/∂β [correction]
        grad_correction = (-(1-beta) / 12 * alpha**2 / f_power_2_minus_2beta + 
                     (1-beta)**2 / 24 * alpha**2 * 2*log_f / f_power_2_minus_2beta +
                     1/4 * rho * alpha * nu / f_power_1_minus_beta * (1 + beta * log_f))*t
        
        return grad_main * correction + main_term * grad_correction
    
    elif param_idx == 2:  # rho

        grad_correction = (1/4 * beta * alpha * nu / f_power_1_minus_beta - rho / 4 * nu**2) * t
        
        return main_term * grad_correction
    
    elif param_idx == 3:  # Volvol

        grad_correction = (1/4 * rho * beta * alpha / f_power_1_minus_beta + (2 - 3*rho**2) / 12 *nu) * t
        
        return main_term * grad_correction

@njit
def compute_general_gradient(param_idx, alpha, beta, rho, nu, K, f, t):
    """
    Calcule le gradient pour la cas général
    """
    # Calcul de z
    z = nu/alpha * (f*K)**((1-beta)/2) * np.log(f/K)
    
    # Calcul de x(z)
    s = np.sqrt(1 - 2*rho*z + z**2)
    x_z = np.log((s + z - rho) / (1 - rho))
    
    # Terme principal
    main_factor = alpha / ((f*K)**((1-beta)/2)) * (z/x_z)
    
    # Terme de correction
    term1 = (1-beta)**2 / 24 * alpha**2 / (f*K)**(1-beta)
    term2 = 1/4 * rho * beta * alpha * nu / ((f*K)**((1-beta)/2))
    term3 = (2 - 3*rho**2) / 24 * nu**2
    
    correction = 1 + (term1 + term2 + term3) * t

    # le terme redondant dans la dérivée du main factor entre tout params
    dx_dz = compute_dx_dz(z, rho)
    red_term = alpha / ((f*K)**((1-beta)/2))/x_z*(1-z/x_z*dx_dz)

    # derivée de z pour tout les params 
    dz_dalpha = -z/alpha
    dz_dbeta = -np.log(f*K)*z/2
    dz_dnu = z/nu
    
    if param_idx == 0:  # Alpha

        # Gradient du terme principal
        grad_main = (z/x_z) / ((f*K)**((1-beta)/2)) + red_term*dz_dalpha
        
        # Gradient de la correction
        grad_correction = ((1-beta)**2 / 12 * alpha / (f*K)**(1-beta) + 
                          1/4 * rho * beta * nu / ((f*K)**((1-beta)/2))) * t
        
        return grad_main * correction + main_factor * grad_correction
    
    elif param_idx == 1:  # ∂σ/∂β

        log_f_k = np.log(f*K)

        grad_main = main_factor*log_f_k/2 + red_term*dz_dbeta

        grad_correction = (-(1-beta) / 12 * alpha**2 / (f*K)**(1-beta) + 
                     (1-beta)**2 / 24 * alpha**2 *log_f_k / (f*K)**(1-beta) +
                     1/4 * rho * alpha * nu / (f*K)**((1-beta)/2) * (1 + beta * log_f_k/2))*t

        return grad_main * correction + main_factor * grad_correction

    elif param_idx == 2:  # Rho
        
        dx_drho = compute_dx_drho(z, rho)
        
        # Gradient du terme principal
        grad_main = -main_factor /x_z * dx_drho
        
        # Gradient de la correction
        grad_correction = ((beta*nu*alpha/(4*((f*K)**((1-beta)/2))))+(-rho / 4 * nu**2)) * t
        
        return grad_main * correction + main_factor * grad_correction
    
    elif param_idx == 3:  # Volvol
        
        # Gradient du terme principal
        grad_main = red_term * dz_dnu
        
        # Gradient de la correction
        grad_correction = (1/4 * rho * beta * alpha / ((f*K)**((1-beta)/2)) + 
                          (2 - 3*rho**2) / 12 *nu) * t
        
        return grad_main * correction + main_factor * grad_correction

@njit
def compute_dx_dz(z, rho):

    s = np.sqrt(1 - 2*rho*z + z**2)

    return 1 / s

@njit
def compute_dx_drho(z, rho):

    s = np.sqrt(1 - 2*rho*z + z**2)
    n = s + z - rho

    return (-z/s-1) / n + 1/(1-rho)
