import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from SABR.prapare_data import prepare_calibration_inputs_K_T
from SABR.Calibrate import calibrated_params_Vol_surface
import mcmc_sabr.config as config
# ===================================== Fonctions de visualisation =====================================

def plot_vol_smile(strike_percentages, maturity, sabr_vols, sabr_vols_vage, market_vols, erreur, erreur_vega):
    """
    Trace le smile de volatilité et sauvegarde le graphique
    
    Args:
        strike_percentages: Pourcentages de strike par rapport au forward
        maturity: Maturité en années
        sabr_vols: Volatilités implicites selon SABR
        market_vols: Volatilités de marché
        beta: Paramètre beta utilisé
        date_part_from_filename: Date extraite du nom de fichier
        use_vega: Booléen indiquant si la pondération par vega a été utilisée
        use_minimize: Booléen indiquant la fonction de minimization utilisée : minimize ou differential_evolution
    """

    # Création du graphique
    colors = plt.cm.viridis(np.array([0,1], dtype='float'))
    plt.figure(figsize=(11, 6))
    plt.title(f'T={round(maturity*365.25)} J')
    plt.plot(strike_percentages, sabr_vols, 'o', color='b', alpha=0.7, label=f'SABR implied vol without vega weighting|error:{erreur:.4f}')
    plt.plot(strike_percentages, sabr_vols_vage, 'o', color='g', alpha=0.7, label=f'SABR implied vol with vega weighting|error:{erreur_vega:.4f}')
    plt.plot(strike_percentages, market_vols, 'o', color='r', alpha=0.7, label='Market vol') 
    plt.legend()
    plt.xlabel('Strike (%)')
    plt.ylabel('Volatility') 
    plt.tight_layout()
    # plt.show()
    
    plt.savefig("results_smile_calibration", transparent=True)


def plot_smile(maturity_index, use_minimize):
    """
    Calcule et visualise les smiles de volatilité pour différentes valeurs de beta à partir du fichier excel
    
    Args:
        use_vega: Booléen indiquant si la pondération par vega doit être utilisée
        use_minimize: Booléen indiquant la fonction de minimization utilisée : minimize ou differential_evolution
        maturity_index: la maturité à analyser
    """

    moneyness, strikes, maturity_data = prepare_calibration_inputs_K_T(config.DATA_FILE_PATH, config.INITIAL_DATE)
    maturities = maturity_data[1:, 0]
    FwdImp = maturity_data[1:, 1]
    market_vols = maturity_data[1:, 2:]
    
    results_scipy = np.zeros((3))
    sabr_vols = np.zeros((len(strikes)))
    sabr_vol_vega = np.zeros((len(strikes)))

    # Extraction des données pour la maturité sélectionnée
    maturity = maturities[maturity_index]
    forward = FwdImp[maturity_index]
    market_vols_for_maturity = market_vols[maturity_index]
    
    calibrated_params, sabr_vols = calibrated_params_Vol_surface(
        maturity, strikes, forward, 
        market_vols_for_maturity, False, 1, use_minimize
    )
    calibrated_params_vega, sabr_vols_vega = calibrated_params_Vol_surface(
        maturity, strikes, forward, 
        market_vols_for_maturity, True, 1, use_minimize
    )

    results_scipy = calibrated_params.x
    erreur = calibrated_params.fun
    nbr_iteration = calibrated_params.nit

    results_scipy_vega = calibrated_params_vega.x
    erreur_vega = calibrated_params_vega.fun
    nbr_iteration_vega = calibrated_params_vega.nit
            
    print(f'without using vega weighting results are: {results_scipy}, and error is: {erreur}, and nbr_iterationis :{nbr_iteration}')
    print(f'using vega weighting results are: {results_scipy_vega}, and error is: {erreur_vega}, and nbr_iterationis :{nbr_iteration_vega}')

    plot_vol_smile(
        moneyness, maturity, sabr_vols, sabr_vols_vega, market_vols_for_maturity, erreur, erreur_vega
    )

def calibrate_sabr(maturity, strikes, forward, market_vols_for_maturity, use_vega, beta, use_minimize):
    results, sabr_vol = calibrated_params_Vol_surface(
            maturity, strikes, forward, 
            market_vols_for_maturity, use_vega, beta, use_minimize
        )
    calibrated_params = results.x
    return calibrated_params
            

def main():
    use_minimize = True
    if "--diff_evo" in sys.argv:
        use_minimize = False
    
    plot_smile(1, use_minimize)


if __name__ == "__main__":
    main()