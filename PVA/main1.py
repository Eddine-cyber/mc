import argparse
import sys
sys.path.append('..') 
import mcmc_sabr.config as config
import numpy as np
from PVA.calculate_pva import plot_pva_results, plot_smiles_distribution
from datetime import datetime

# Importation dynamique des classes MCMC
from mcmc_sabr.mcmc_vectorized import MCMCVectorized
from mcmc_sabr.mcmc_beta_fixed import MCMCBetaFixed
from mcmc_sabr.mcmc_mala import MCMCMala

ALGO_MAPPING = {
    'vectorized': MCMCVectorized,
    'beta_fixed': MCMCBetaFixed,
    'mala': MCMCMala,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default=config.DEFAULT_ALGO, choices=ALGO_MAPPING.keys())
    parser.add_argument('--n_iterations', type=int, default=config.N_ITERATIONS)
    parser.add_argument('--init', type=str, default=config.INITIALIZATION_METHOD, choices=['calibrate', 'manual'])
    parser.add_argument('--no_diagnostics', dest='diagnostics', action='store_false')
    parser.add_argument('--no_adapt_cov', dest='adapt_cov', action='store_false')
    
    args = parser.parse_args()
    
    # ===========================================================================
    # # Convertir les arguments en dictionnaire
    # cli_args = vars(args)

    # for j in [0] :#range(10):
    #     config.MATURITY_INDEX = j
    #     mcmc_class = ALGO_MAPPING[cli_args['algo']]
    #     sampler = mcmc_class(cli_args)
    #     samples = sampler.run_sampler()["samples"][0] # car on veut acceder que pour une seul chaine

    #     maturity, f, market_vols, strikes = sampler.market_data

    #     s_0 = strikes[8]
    #     interest_rate = 1/maturity*np.log(f/s_0)  # strikes[8] = S_0
    #     option = ["call", "put"]
    #     for opt in range(1):
    #         for i in [0, 4, 8, 10, 12]:#[0,2,4,6,8,10,12]:
    #             strike = strikes[i]
    #             implied_vol = market_vols[i]
    #             strike_percentage = round(strike/s_0*100)
    #             if strike_percentage > 100:
    #                 if opt == 0:
    #                     moneyness = "OTM"
    #                 else :
    #                     moneyness = "ITM"
    #             elif strike_percentage < 100:
    #                 if opt == 0:
    #                     moneyness = "ITM"
    #                 else :
    #                     moneyness = "OTM"
    #             else:
    #                 moneyness = "ATM"

    #             print(f"[{option[opt]} k={strike_percentage}% T={round(maturity*365.25)} Froward_at_maturity={f}] interest rate={interest_rate}")
    #             # pva, ref_price, percentile_10, prices, weights, sorted_prices, cumsum_weights = calculate_pva_weighted(sampler, option[opt], samples, f, maturity, strike, implied_vol, interest_rate)
    #             # print(f"  PVA/Prix ratio: {pva/ref_price*100:.1f}%")

    #             fig, results = plot_pva_results(sampler, option[opt], samples, f, maturity, strike, implied_vol, s_0, interest_rate)
    #             import os
    #             save_path = f"Results_{cli_args['algo']}_{option[opt]}/{moneyness}/pva_resultats_{j}"
    #             full_file_path = f"{save_path}/{strike_percentage}.png"
    #             os.makedirs(save_path, exist_ok=True)
    #             # Sauvegarder la figure
    #             fig.savefig(full_file_path)
    # ===========================================================================

    # Convertir les arguments en dictionnaire
    cli_args = vars(args)
    mcmc_class = ALGO_MAPPING[cli_args['algo']]
    sampler = mcmc_class(cli_args)
    samples = sampler.run_sampler()["samples"][0]
    maturity, f, market_vols, strikes = sampler.market_data

    s_0 = strikes[8] # strikes[8] = S_0
    strike_percentages = np.array([round(strike/s_0*100) for strike in strikes])

    plot_smiles_distribution(sampler, samples, f, maturity, strikes, strike_percentages, market_vols, 'one maturity')

if __name__ == "__main__":
    main()

