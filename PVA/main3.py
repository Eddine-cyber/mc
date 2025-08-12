import argparse
import sys
sys.path.append('..') 
import mcmc_sabr.config as config
import numpy as np
from PVA.calculate_pva import plot_pva_results, plot_smiles_distribution
from datetime import datetime

# Importation dynamique des classes MCMC
from mcmc_sabr_3.mcmc_vectorized import MCMCVectorized
from mcmc_sabr_3.mcmc_beta_fixed import MCMCBetaFixed

ALGO_MAPPING = {
    'vectorized': MCMCVectorized,
    'beta_fixed': MCMCBetaFixed,
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
    # mcmc_class = ALGO_MAPPING[cli_args['algo']]
    # sampler = mcmc_class(cli_args)
    # samples = sampler.run_sampler()["samples"][0]
    # maturities, FwdImp, market_vols, strikes = sampler.market_data_all_maturities

    # ratio_pva = np.zeros((10, 13))
    # for j in [0, 6, 8] :#range(10):
    #     maturity, f = maturities[j], FwdImp[j]

    #     s_0 = strikes[8] # strikes[8] = S_0
    #     interest_rate = 1/maturity*np.log(f/s_0)  
    #     option = ["call", "put"]
    #     for opt in range(1):
    #         for i in [0,8,12]:#range(13):
    #             strike = strikes[i]
    #             implied_vol = market_vols[j, i]
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

    #             fig, results = plot_pva_results(sampler, option[opt], samples, f, maturity, strike, implied_vol, s_0, interest_rate)
    #             pva, ref_price = results["pva"], results["ref_price"]
    #             print(f"  PVA/Prix ratio: {pva/ref_price*100:.1f}%")
    #             ratio_pva[j,i] = pva/ref_price*100
    #             import os
    #             save_path = f"Results_beta_{config.FIXED_BETA}_{cli_args['algo']}_{option[opt]}/{moneyness}/pva_resultats_{j}"
    #             full_file_path = f"{save_path}/{strike_percentage}.png"
    #             os.makedirs(save_path, exist_ok=True)
    #             # Sauvegarder la figure
    #             fig.savefig(full_file_path)

    # print("ratio PVA \n :", ratio_pva)
    # ===========================================================================

    # Convertir les arguments en dictionnaire
    cli_args = vars(args)
    mcmc_class = ALGO_MAPPING[cli_args['algo']]
    sampler = mcmc_class(cli_args)
    samples = sampler.run_sampler()["samples"][0]
    maturity, f, market_vols, strikes = sampler.market_data

    s_0 = strikes[8] # strikes[8] = S_0
    strike_percentages = np.array([round(strike/s_0*100) for strike in strikes])

    plot_smiles_distribution(sampler, samples, f, maturity, strikes, strike_percentages, market_vols, 'one maturity-multi valuation dates')

if __name__ == "__main__":
    main()

