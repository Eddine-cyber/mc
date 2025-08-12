import argparse
import sys
sys.path.append('..') 
import mcmc_sabr.config as config

from mcmc_sabr_3.mcmc_vectorized import MCMCVectorized
from mcmc_sabr_3.mcmc_beta_fixed import MCMCBetaFixed
from mcmc_sabr.mcmc_mala import MCMCMala
from mcmc_sabr.mcmc_sequential import MCMCSequential
from mcmc_sabr.mcmc_mala_beta_fixed import MCMCMalaBetaFixed

ALGO_MAPPING = {
    'vectorized': MCMCVectorized,
    'beta_fixed': MCMCBetaFixed,
    'mala': MCMCMala,
    'sequential': MCMCSequential,
    'mala_beta_fixed': MCMCMalaBetaFixed,
}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default=config.DEFAULT_ALGO, choices=ALGO_MAPPING.keys())
    parser.add_argument('--n_iterations', type=int, default=config.N_ITERATIONS)
    parser.add_argument('--chains', type=int, default=config.N_CHAINS)
    parser.add_argument('--init', type=str, default=config.INITIALIZATION_METHOD, choices=['calibrate', 'manual1', 'manual2'])
    parser.add_argument('--no_diagnostics', dest='diagnostics', action='store_false')
    parser.add_argument('--no_adapt_cov', dest='adapt_cov', action='store_false')
    parser.add_argument('--maturity', type=int, default=config.MATURITY_INDEX, choices=[i for i in range(10)])

    
    args = parser.parse_args()
    
    # Convertir les arguments en dictionnaire
    cli_args = vars(args)

    mcmc_class = ALGO_MAPPING[cli_args['algo']]
    sampler = mcmc_class(cli_args)
    sampler.run_sampler()
 
if __name__ == "__main__":
    main()