from mcmc_sabr_2.mcmc_beta_fixed_2 import MCMCBetaFixed
from mcmc_sabr_2.mcmc_mala_2 import MCMCMala
import mcmc_sabr.config as config


class MCMCMalaBetaFixed(MCMCBetaFixed, MCMCMala):
    """Implémentation du MCMC avec gradient (MALA) en utilisant le gradient analytique."""

    def _initialize_sampler(self):        
        super()._initialize_sampler()
        self.langevin_error = config.LANGEVIN_ERROR
        print("Initialized MALA MCMC with analytical gradient.")    