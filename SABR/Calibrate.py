import numpy as np
import scipy.optimize as sc
from SABR.sabr import vol_sabr_hagan, vega
import mcmc_sabr.config as config
# ===================================== Fonctions de calibration =====================================

def sabr_objective_function(params, beta, strikes, market_vols, forward, maturity, use_vega):
    """
    Fonction objectif pour la minimisation (retourne la somme des carrés d'erreur)
    
    Args:
        params: Paramètres du modèle SABR [alpha, rho, volvol]
        beta: Paramètre beta fixé
        strikes: Prix d'exercice
        market_vols: Volatilités de marché
        forward: Prix forward
        maturity: Maturité en années
        use_vega: Booléen indiquant si la pondération par vega doit être utilisée
        
    Returns:
        Erreur quadratique (pondérée ou non par vega)
    """

    alpha, rho, volvol = params[0], params[1], params[2]
    sabr_vols = np.zeros_like(market_vols)
    sabr_vols = vol_sabr_hagan(strikes, forward, maturity, alpha, beta, rho, volvol)
    # Supprimer les NaN potentiels pour calculer l'erreur
    valid_mask = ~np.isnan(sabr_vols)
    
    if not use_vega:
        error = np.sum((sabr_vols[valid_mask] - market_vols[valid_mask])**2)
    else:
        # Calcul de vega pour la pondération
        vegas = vega(forward, strikes, maturity, market_vols)
        error = np.sum(vegas[valid_mask] * ((sabr_vols[valid_mask] - market_vols[valid_mask])**2))
    return error


def calibrate_sabr_hagan_with_scipy(beta, strikes, market_vols, forward, maturity, use_vega, use_minimize):
    """
    Calibration du modèle SABR en utilisant scipy.optimize.minimize et scipy.optimize.differential_evolution
    pour scipy.optimize.differential_evolution: on a pas besoin de spécifier les initial_guess des params
    
    Args:
        beta: Paramètre beta fixé
        strikes: Prix d'exercice
        market_vols: Volatilités de marché
        forward: Prix forward
        maturity: Maturité en années
        use_vega: Booléen indiquant si la pondération par vega doit être utilisée
        use_minimize: Booléen indiquant la fonction de minimization utilisée : minimize ou differential_evolution
        initial_guess: Valeurs initiales pour [alpha, rho, volvol] pour scipy.optimize.minimize
        bounds: Contraintes de bornes pour les paramètres 
        
    Returns:
        Résultat de l'optimisation
    """
    bounds_err = 0.001
    bounds = [(bounds_err, 10.0), (-1.0+bounds_err, 1.0-bounds_err), (bounds_err, 10.0)]

    args = (beta, strikes, market_vols, forward, maturity, use_vega)
    if use_minimize == False:
        result = sc.differential_evolution(sabr_objective_function, args=args, bounds=bounds, tol=0.001)
    else:
        initial_guess = [0.2, -0.4, 1.0]
        result = sc.minimize(sabr_objective_function, initial_guess, args=args, method='L-BFGS-B', bounds=bounds)
    return result

# ===================================== Fonctions de calibration et génération de surfaces =====================================

def calibrated_params_Vol_surface(maturitie, strikes, forward, market_vols, use_vega, beta, use_minimize):
    """
    Calibre les paramètres du modèle SABR et génère la surface de volatilité
    
    Args:
        maturitie: Maturité en années
        strikes: Prix d'exercice
        strike_percentages: Pourcentages de strike par rapport au forward
        forward: Prix forward
        market_vols: Volatilités de marché
        excel_file: Chemin vers le fichier Excel
        use_vega: Booléen indiquant si la pondération par vega doit être utilisée
        use_minimize: Booléen indiquant la fonction de minimization utilisée : minimize ou differential_evolution
        beta: Paramètre beta fixé
        
    Returns:
        Tuple (résultats de calibration, volatilités SABR)
    """
    results_scipy = calibrate_sabr_hagan_with_scipy(beta, strikes, market_vols, forward, maturitie, use_vega, use_minimize)
    shape_sabr_vols = market_vols.shape
    sabr_vols = SABR_implied_vol_surface(results_scipy.x, beta, strikes, shape_sabr_vols, forward, maturitie)
        
    return results_scipy, sabr_vols

def calibrate_sabr(maturity, strikes, forward, market_vols_for_maturity, use_vega, beta, use_minimize):
    results, sabr_vol = calibrated_params_Vol_surface(
            maturity, strikes, forward, 
            market_vols_for_maturity, use_vega, beta, use_minimize
        )
    calibrated_params = results.x
    print("erreur final de la fonction objectif :", results.fun)
    return calibrated_params

def calibrate_sabr_with_beta(maturity, strikes, forward, market_vols, use_vega=True, use_minimize=True):
    results = calibrate_sabr_hagan_with_scipy_with_beta(strikes, market_vols, forward, maturity, use_vega, use_minimize)
    return results.x

def calibrate_sabr_hagan_with_scipy_with_beta(strikes, market_vols, forward, maturity, use_vega, use_minimize):
    bounds = list(zip(config.FOUR_PARAM_BOUNDS[0], config.FOUR_PARAM_BOUNDS[1]))
    args = (strikes, market_vols, forward, maturity, use_vega)
    if use_minimize == False:
        result = sc.differential_evolution(sabr_objective_function_with_beta, args=args, bounds=bounds, tol=0.001)
    else:
        initial_guess = [0.2, 0.94, -0.4, 1.0]
        result = sc.minimize(sabr_objective_function_with_beta, initial_guess, args=args, method='L-BFGS-B', bounds=bounds)
    return result

def sabr_objective_function_with_beta(params, strikes, market_vols, forward, maturity, use_vega):

    alpha, beta, rho, volvol = params[0], params[1], params[2], params[3]
    sabr_vols = np.zeros_like(market_vols)
    sabr_vols = vol_sabr_hagan(strikes, forward, maturity, alpha, beta, rho, volvol)
    # Supprimer les NaN potentiels pour calculer l'erreur
    valid_mask = ~np.isnan(sabr_vols)
    
    if not use_vega:
        error = np.sum((sabr_vols[valid_mask] - market_vols[valid_mask])**2)
    else:
        # Calcul de vega pour la pondération
        vegas = vega(forward, strikes, maturity, market_vols)
        error = np.sum(vegas[valid_mask] * ((sabr_vols[valid_mask] - market_vols[valid_mask])**2))
    return error



def SABR_implied_vol_surface(params, beta, strikes, shape_sabr_vols, forward, maturitie):
    """
    Calcule la surface de volatilité implicite selon le modèle SABR
    
    Args:
        params: Paramètres du modèle [alpha, rho, volvol]
        beta: Paramètre beta fixé
        strikes: Prix d'exercice
        shape_sabr_vols: pour la forme du résultat (sabr_vols)
        forward: Prix forward
        maturitie: Maturité en années
        
    Returns:
        Volatilités implicites selon SABR
    """
    alpha, rho, volvol = params[0], params[1], params[2]
    sabr_vols = np.zeros(shape_sabr_vols)
    sabr_vols  = vol_sabr_hagan(strikes, forward, maturitie, alpha, beta, rho, volvol)
    return sabr_vols


def calibrate_sabr_all_maturities(strikes, market_vols, forwards, maturities):
    bounds = list(zip(config.FOUR_PARAM_BOUNDS[0], config.FOUR_PARAM_BOUNDS[1]))
    args = (strikes, market_vols, forwards, maturities)

    initial_guess = [0.2, 0.94, -0.4, 1.0]
    result = sc.minimize(sabr_objective_function_all_maturities, initial_guess, args=args, method='L-BFGS-B', bounds=bounds)
    return result.x

def sabr_objective_function_all_maturities(params, strikes, market_vols, forwards, maturities):
    alpha, beta, rho, volvol = params[0], params[1], params[2], params[3]
    sabr_vols = np.zeros_like(market_vols)
    error = 0.0
    for i in range(len(maturities)):
        forward, maturity, market_vols_maturity = forwards[i], maturities[i], market_vols[i]
        sabr_vols = vol_sabr_hagan(strikes, forward, maturity, alpha, beta, rho, volvol)
        # Supprimer les NaN potentiels pour calculer l'erreur
        valid_mask = ~np.isnan(sabr_vols)
        vegas = vega(forward, strikes, maturity, market_vols_maturity)
        error += np.sum(vegas[valid_mask] * ((sabr_vols[valid_mask] - market_vols_maturity[valid_mask])**2))
    return error

def calibrate_sabr_all_maturities_without_beta(strikes, market_vols, FwdImp, maturities, beta) :
    bounds = list(zip(config.THREE_PARAM_BOUNDS[0], config.THREE_PARAM_BOUNDS[1]))
    args = (strikes, market_vols, FwdImp, maturities, beta)

    initial_guess = [0.2, -0.4, 1.0]
    result = sc.minimize(sabr_objective_function_all_maturities_without_beta, initial_guess, args=args, method='L-BFGS-B', bounds=bounds)
    print("erreur final de la fonction objectif :", result.fun)
    return result.x

def sabr_objective_function_all_maturities_without_beta(params, strikes, market_vols, forwards, maturities, beta):
    alpha, rho, volvol = params[0], params[1], params[2]
    sabr_vols = np.zeros_like(market_vols)
    error = 0.0
    for i in range(len(maturities)):
        forward, maturity, market_vols_maturity = forwards[i], maturities[i], market_vols[i]
        sabr_vols = vol_sabr_hagan(strikes, forward, maturity, alpha, beta, rho, volvol)
        # Supprimer les NaN potentiels pour calculer l'erreur
        valid_mask = ~np.isnan(sabr_vols)
        vegas = vega(forward, strikes, maturity, market_vols_maturity)
        error += np.sum(vegas[valid_mask] * ((sabr_vols[valid_mask] - market_vols_maturity[valid_mask])**2))
    return error