import numpy as np
from scipy.stats import norm
from numba import njit


def calculate_vanilla_price(option, sigma, f, k, maturity, interest_rate):
    
    discount_factor = np.exp(-interest_rate * maturity)
    
    d1 = (np.log(f/k) + (sigma**2/2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    
    c = (f * norm.cdf(d1) - k * norm.cdf(d2)) * discount_factor
    
    if option == "call":
        return c
    else:
        return c + discount_factor * (k - f)
    


def vega(F, K, tau, sigma):
    """
    Calcule le vega d'une option selon le modèle Black-Scholes.
    
    Args:
        F: Prix forward
        K: Prix d'exercice
        tau: Temps jusqu'à l'échéance (T-t)
        sigma: Volatilité
    
    Returns:
        La valeur du vega
    """
    # Calcul de d1 selon la formule de Black-Scholes
    d1 = (np.log(F/K) + ((sigma**2)/2) * tau) / (sigma * np.sqrt(tau))
    
    # Calcul du vega
    vega = F * norm.pdf(d1) * np.sqrt(tau)
    
    return vega


def vol_sabr_hagan_one_strike(k, f, t, alpha, beta, rho, volvol, epsilon=1):
    """
    Calcule la volatilité implicite selon l'approximation de Hagan pour le modèle SABR.
    
    Args:
        K: Prix d'exercice (array)
        f: Prix forward actuel
        t: Temps jusqu'à l'échéance
        alpha: Paramètre alpha du modèle
        beta: Paramètre beta du modèle
        rho: Corrélation
        volvol: Volatilité de la volatilité
        epsilon: Facteur d'approximation (défaut: 1)
    
    Returns:
        Tuple contenant:
        - Volatilité implicite pour chaque prix d'exercice
        - Facteur de correction temporelle (utiliser pour le debug)
    """

    # =============  Traitement pour K != f  =============
    if k != f:
        # Calculs préliminaires
        logfk_k_different_f = np.log(f / k)
        fk_pow_half = (f * k)**((1 - beta) / 2.0)
        fk_pow_full = fk_pow_half**2  # (f*K)**(1-beta)
        first_term = alpha / fk_pow_half

        # Calcul du terme de correction temporelle
        term1_t = ((1 - beta)**2 / 24.0) * (alpha**2 / fk_pow_full)
        term2_t = 0.25 * rho * beta * volvol * alpha / fk_pow_half
        term3_t = ((2.0 - 3.0 * rho**2) / 24.0) * (volvol**2)
        time_correction_factor = 1.0 + (term1_t + term2_t + term3_t) * t * epsilon

        # Calcul de z (Eq 2.17b)
        z = (volvol / alpha) * fk_pow_half * logfk_k_different_f

        # Calcul de x(z) (Eq 2.17c)
        term_xz = 1.0 - 2.0 * rho * z + z**2
        sqrt_term_xz = np.sqrt(term_xz)
        numerator_xz = sqrt_term_xz + z - rho
        denominator_xz = 1.0 - rho
        
        # Calcul de log(numerator_xz / denominator_xz)
        log_arg_xz = numerator_xz / denominator_xz
        xz = np.log(log_arg_xz)
        
        # Facteur z / x(z)
        z_over_xz = z / xz

        # Calcul du dénominateur (série logarithmique)
        logfk_sq = logfk_k_different_f**2
        logfk_pow4 = logfk_k_different_f**4
        term_log2 = (((1 - beta)**2) / 24.0) * logfk_sq
        term_log4 = (((1 - beta)**4) / 1920.0) * logfk_pow4
        denominator_log_series = 1.0 + term_log2 + term_log4

        # Calcul de la volatilité implicite pour K != f
        sigma_B_without_f = (first_term / denominator_log_series) * z_over_xz * time_correction_factor
        return sigma_B_without_f
        
    # =============  Traitement pour K == f  =============
    if k == f:
        fk_pow_half_f = (f * k)**((1 - beta) / 2.0)
        fk_pow_full_f = fk_pow_half_f**2
        first_term_f = alpha / fk_pow_half_f

        # Calcul du terme de correction temporelle pour K == f
        term1_t_f = ((1 - beta)**2 / 24.0) * (alpha**2 / fk_pow_full_f)
        term2_t_f = 0.25 * rho * beta * volvol * alpha / fk_pow_half_f
        term3_t_f = ((2.0 - 3.0 * rho**2) / 24.0) * (volvol**2)
        time_correction_factor_f = 1.0 + (term1_t_f + term2_t_f + term3_t_f) * t * epsilon
        
        # Formule simplifiée pour K == f (car z/x(z) = 1 quand K = f)
        sigma_B_f = first_term_f * time_correction_factor_f
        return sigma_B_f

@njit
def vol_sabr_hagan_ATM(f, t, alpha, beta, rho, volvol, epsilon=1):

    fk_pow_half_f = f**(1 - beta)
    fk_pow_full_f = fk_pow_half_f**2
    first_term_f = alpha / fk_pow_half_f

    # Calcul du terme de correction temporelle pour K == f
    term1_t_f = ((1 - beta)**2 / 24.0) * (alpha**2 / fk_pow_full_f)
    term2_t_f = 0.25 * rho * beta * volvol * alpha / fk_pow_half_f
    term3_t_f = ((2.0 - 3.0 * rho**2) / 24.0) * (volvol**2)
    time_correction_factor_f = 1.0 + (term1_t_f + term2_t_f + term3_t_f) * t * epsilon
    
    # Formule simplifiée pour K == f (car z/x(z) = 1 quand K = f)
    sigma_B_f = first_term_f * time_correction_factor_f
    return sigma_B_f

@njit
def vol_sabr_hagan_Not_ATM(strikes, f, t, alpha, beta, rho, volvol, epsilon=1):

    # Calculs préliminaires
    logfk_k_different_f = np.log(f / strikes)
    fk_pow_half = (f * strikes)**((1 - beta) / 2.0)
    fk_pow_full = fk_pow_half**2  # (f*K)**(1-beta)
    first_term = alpha / fk_pow_half

    # Calcul du terme de correction temporelle
    term1_t = ((1 - beta)**2 / 24.0) * (alpha**2 / fk_pow_full)
    term2_t = 0.25 * rho * beta * volvol * alpha / fk_pow_half
    term3_t = ((2.0 - 3.0 * rho**2) / 24.0) * (volvol**2)
    time_correction_factor = 1.0 + (term1_t + term2_t + term3_t) * t * epsilon

    # Calcul de z (Eq 2.17b)
    z = (volvol / alpha) * fk_pow_half * logfk_k_different_f

    # Calcul de x(z) (Eq 2.17c)
    term_xz = 1.0 - 2.0 * rho * z + z**2
    sqrt_term_xz = np.sqrt(term_xz)
    numerator_xz = sqrt_term_xz + z - rho
    denominator_xz = 1.0 - rho
    
    # Calcul de log(numerator_xz / denominator_xz)
    log_arg_xz = numerator_xz / denominator_xz
    xz = np.log(log_arg_xz)
    
    # Facteur z / x(z)
    z_over_xz = z / xz

    # Calcul du dénominateur (série logarithmique)
    logfk_sq = logfk_k_different_f**2
    logfk_pow4 = logfk_k_different_f**4
    term_log2 = (((1 - beta)**2) / 24.0) * logfk_sq
    term_log4 = (((1 - beta)**4) / 1920.0) * logfk_pow4
    denominator_log_series = 1.0 + term_log2 + term_log4

    # Calcul de la volatilité implicite pour K != f
    sigma_B_without_f = (first_term / denominator_log_series) * z_over_xz * time_correction_factor
    return sigma_B_without_f



def vol_sabr_hagan(strikes, f, t, alpha, beta, rho, volvol, epsilon=1):
    """
    Calcule la volatilité implicite selon l'approximation de Hagan pour le modèle SABR.
    
    Args:
        strikes: Prix d'exercice (array)
        f: Prix forward actuel
        t: Temps jusqu'à l'échéance
        alpha: Paramètre alpha du modèle
        beta: Paramètre beta du modèle
        rho: Corrélation
        volvol: Volatilité de la volatilité
        epsilon: Facteur d'approximation (défaut: 1)
    
    Returns:
        Tuple contenant:
        - Volatilité implicite pour chaque prix d'exercice
        - Facteur de correction temporelle (utiliser pour le debug)
    """

    mask = np.isclose(strikes, f, atol=1e-6)
    exists = np.any(mask)

    if exists:
        # Trouver l'indice où K == f
        indice = np.where(mask)[0]
        # =============  Traitement pour K != f  =============
        strikes_without_f = np.concatenate((strikes[:indice], strikes[indice+1:]))
        sigma_B_without_f = vol_sabr_hagan_Not_ATM(strikes_without_f, f, t, alpha, beta, rho, volvol, epsilon)
        # =============  Traitement pour K == f  =============
        sigma_B_f = vol_sabr_hagan_ATM(f, t, alpha, beta, rho, volvol, epsilon)
        # Combinaison des résultats
        sigma_B = np.concatenate((sigma_B_without_f[:indice], sigma_B_f, sigma_B_without_f[indice:]))
    else:
        sigma_B = vol_sabr_hagan_Not_ATM(strikes, f, t, alpha, beta, rho, volvol, epsilon)
    
    return sigma_B
  
