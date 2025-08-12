import numpy as np
import time
from abc import ABC, abstractmethod
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt


import mcmc_sabr.config as config
from mcmc_sabr.utils import calculate_weights_likelihood, calculate_weights, calculate_prior_mean, load_market_data_multi_valuation_dates, get_initial_params_from_calibration,  compute_prior_mean_std, compute_covariance_matrix_from_hessienne
from mcmc_sabr.diagnostics import calculate_r_hat
from mcmc_sabr.html_report import save_results_to_html
from SABR.sabr import vol_sabr_hagan
from SABR.prapare_data import prepare_calibration_inputs
from mcmc_sabr.utils import _inverse_transform_4_params, _inverse_transform_3_params, _transform_4_params, _transform_3_params

class MCMCBase(ABC):
    """
    Abstract base class for MCMC samplers.
    This class provides the common framework and functionality for different MCMC
    sampling strategies .
    
    Attributes :
    n_iterations : int
        Total number of MCMC iterations to perform.
    n_chains : int
        Number of independent MCMC chains to run in parallel.
    adapt_cov : bool
        Whether to use adaptive covariance for proposal distribution.
    diagnostics_enabled : bool
        Whether to compute and save diagnostic statistics.
    init_method : str
        Method for initializing parameters ('calibrate' or 'manual').
    burn_in : int
        Number of burn-in iterations.
    n0 : int
        Batch size for adaptive covariance updates.
    epsilon_cov : float
        Small positive constant for numerical stability in covariance updates.
    param_names : list
        Names of model parameters (set in subclasses).
    n_params : int
        Number of parameters (3 or 4 depending on subclass).
    params_bounds : tuple
        Parameter bounds (min, max) arrays.
    s_d : float
        Scaling factor for proposal distribution.
    market_data : tuple
        Market data (maturity, forward, vols, strikes).
    initial_params : np.ndarray
        Initial parameter values for each chain.
    prior_means : np.ndarray
        Prior distribution means for each chain.
    prior_covs : np.ndarray
        Prior covariance matrices for each chain.
    cov_proposal : np.ndarray
        Proposal covariance matrices for each chain.
    mean_proposal : np.ndarray
        Proposal means for adaptive sampling.
    new_samples_buffer : np.ndarray
        Buffer for storing samples in adaptive covariance.
    """

    def __init__(self, cli_args):
        """
        Initialize the MCMC sampler with command line arguments.
        
        Parameters:
        cli_args : dict
            Dictionary containing command line arguments:
            - n_iterations: Number of MCMC iterations
            - chains: Number of parallel chains
            - adapt_cov: Whether to use adaptive covariance
            - diagnostics: Whether to enable diagnostics
            - init: Initialization method ('calibrate' or 'manual')
        """
        self.n_iterations = cli_args.get('n_iterations', config.N_ITERATIONS)
        self.n_chains = cli_args.get('chains', config.N_CHAINS)
        self.adapt_cov = cli_args.get('adapt_cov', config.ADAPTIVE_COVARIANCE)
        self.diagnostics_enabled = cli_args.get('diagnostics', config.DIAGNOSTICS_ENABLED)
        self.init_method = cli_args.get('init', config.INITIALIZATION_METHOD)

        self.burn_in = self.n_iterations // 10 if not self.adapt_cov else self.n_iterations // 4
        self.n0 = int(np.sqrt(self.n_iterations))
        self.epsilon_cov = 1e-8

        # Implementation-specific attributes (defined in subclasses)
        self.param_names = []
        self.n_params = 0
        self.params_bounds = None
        self.s_d = 0

        # Data
        self.market_data = None
        self.initial_params = None
        self.prior_means = None
        self.prior_covs = None
        self.cov_proposal = None
        self.mean_proposal = None
        self.new_samples_buffer = None
        self.maturity = None
        self.weight = None
        self.likelihood_weight = None

    @abstractmethod
    def _initialize_sampler(self):
        """
        Initialize attributes specific to the MCMC variant.
        
        This method must be implemented by subclasses to set up:
        - param_names: List of parameter names
        - n_params: Number of parameters (3 or 4)
        - params_bounds: Parameter bounds
        - s_d: Scaling factor for proposals
        """
        pass

    @abstractmethod
    def transform_parameter(self, param_value):
        """
        Transform parameters from original to unconstrained space.
        """
        pass

    @abstractmethod
    def inverse_transform_parameter(self, transformed_value):
        """
        Transform parameters from unconstrained back to original space.
        """
        pass

    @abstractmethod
    def compute_jacobian_det(self, param_value):
        """
        Compute the Jacobian determinant for parameter transformation.
        """
        pass

    @abstractmethod
    def _run_one_iteration(self, iteration):
        """
        Execute a single iteration of the sampler.
        """
        pass

    def setup(self):
        """
        Prepare the sampler for execution.
        
        This method performs all necessary initialization:
        1. Initializes sampler-specific attributes
        2. Loads market data
        3. Sets initial parameter values
        4. Configures prior distributions
        5. Initializes proposal distributions
        """
        print(f"--- Setting up {self.__class__.__name__} Sampler ---")
        self._initialize_sampler()

        self.market_data_multi_valuation_dates = load_market_data_multi_valuation_dates()
        maturity, forward_all_valuation_dates, market_vols_all_valuation_dates, strikes_all_valuation_dates = self.market_data_multi_valuation_dates
        f, market_vols, strikes = forward_all_valuation_dates[0], market_vols_all_valuation_dates[0], strikes_all_valuation_dates[0]
        self.market_data = maturity, f, market_vols, strikes

        self.maturity = maturity
        self.weight = calculate_weights()
        self.likelihood_weight = calculate_weights_likelihood()
        print("weights:", self.weight)
        print("weights for likelihood:", self.likelihood_weight)
        # Parameter initialization
        if self.init_method == 'calibrate':
            print("=== calibrating SABR to get initial params ===")
            print("="*50)
            calibrated_params = []
            for i in range(len(strikes_all_valuation_dates)):
                data_market = maturity, forward_all_valuation_dates[i], market_vols_all_valuation_dates[i], strikes_all_valuation_dates[i]
                calibrated_params.append(get_initial_params_from_calibration(data_market, self.n_params, self.n_chains)) 
                print(f"calibrated params for valuation date {config.INITIAL_DATES[i]} and maturity {maturity} are:")
                print(calibrated_params[i])
            print("="*50)
            self.initial_params = calculate_prior_mean(calibrated_params, self.weight)
            print(f"averaged calibrated param for maturity {maturity} is :")
            print(self.initial_params)
            
        elif self.init_method == 'manual1':
            self.initial_params = config.MANUAL_INITIAL_PARAMS_4 if self.n_params == 4 else config.MANUAL_INITIAL_PARAMS_3
            self.initial_params = self.initial_params[:self.n_chains]
        elif self.init_method == 'manual2':
            self.initial_params = config.MANUAL2_INITIAL_PARAMS_4 if self.n_params == 4 else config.MANUAL2_INITIAL_PARAMS_3
            self.initial_params = self.initial_params[:self.n_chains]
        print("Initial parameters:\n", self.initial_params)



        # Prior initialization
        # prior_bounds = config.PRIOR_BOUNDS_4_PARAMS if self.n_params == 4 else config.PRIOR_BOUNDS_3_PARAMS
        # mean, std = compute_prior_mean_std(config.PRIOR_CONFIDENCE_INTERVAL, prior_bounds, self.transform_parameter)
        # self.prior_means = np.tile(mean, (self.n_chains, 1))
        # prior_cov = np.diag(std**2)
        
        mean = _transform_4_params(self.initial_params[0]) if self.n_params == 4 else _transform_3_params(self.initial_params[0])
        self.prior_means = np.tile(mean, (self.n_chains, 1))
        prior_cov = np.diag(np.array([0.5, 0.5, 0.5, 0.5])) if self.n_params == 4 else np.diag(np.array([0.5, 0.1, 0.9]))
        self.prior_covs = np.tile(prior_cov, (self.n_chains, 1, 1))

        self.prior_transformed_means = _inverse_transform_4_params(mean) if self.n_params == 4 else _inverse_transform_3_params(mean)
        print("prior means")
        print(self.prior_transformed_means)
        print("prior covs")
        print(self.prior_covs)

        # Proposal initialization
        if self.adapt_cov :
            self.cov_proposal = self.prior_covs.copy()
            self.mean_proposal = self.prior_means.copy()
            self.new_samples_buffer = np.zeros((self.n_chains, self.n0, self.n_params))
        else :
            # t_maturity, forward, _, strikes = self.market_data
            strikes, maturity_data = prepare_calibration_inputs(config.DATA_FILE_PATH, config.INITIAL_DATE)
            maturities = maturity_data[1:, 0]
            FwdImp = maturity_data[1:, 1]
            # market_vols = maturity_data[1:, 2:]

            covariance_matrix = compute_covariance_matrix_from_hessienne(self.initial_params[0], strikes, FwdImp, maturities)
            self.cov_proposal = np.tile(covariance_matrix, (self.n_chains, 1, 1))

        self.current_params = self.initial_params.copy()
        print("Setup complete.")

    def run_sampler(self):
        """
        Execute the main MCMC loop.
        
        Returns
        -------
        dict
            Results dictionary containing:
            - samples: Array of parameter samples after burn-in
            - acceptance_rates: Acceptance rate for each chain
            - param_ranges: Minimum and maximum parameter values observed
            - r_hat: Gelman-Rubin convergence diagnostic
        """
        start_time = time.time()
        print(f"\n--- Running MCMC Sampler for {self.n_iterations} iterations ---")
        
        self.setup()

        n_samples_after_burn_in = self.n_iterations - self.burn_in
        samples = np.zeros((self.n_chains, n_samples_after_burn_in, self.n_params))
        self.acceptance_counts = np.zeros(self.n_chains)
        min_params = self.initial_params.copy()
        max_params = self.initial_params.copy()

        for iteration in tqdm(range(self.n_iterations), desc="MCMC Progress"):
            self._run_one_iteration(iteration)

            if iteration >= self.burn_in:
                samples[:, iteration - self.burn_in] = self.current_params
            
            min_params = np.minimum(min_params, self.current_params)
            max_params = np.maximum(max_params, self.current_params)

        acceptance_rates = (self.acceptance_counts * 100) / self.n_iterations

        elapsed_time = time.time() - start_time
        print(f"\n--- MCMC run finished in {elapsed_time:.2f} seconds with acceptance rates: {acceptance_rates} ---")
        print("====================  correlation_matrix is  ====================")
        correlation_matrix = np.corrcoef(samples[0].T)
        print(correlation_matrix)

        r_hat = calculate_r_hat(samples)

        results = {
            'samples': samples,
            'acceptance_rates': acceptance_rates,
            'param_ranges': (min_params, max_params),
            'r_hat': r_hat
        }
        print(f"\n--- begin generate report ---")
        self.generate_report(results)
        print(f"\n--- generate report finished ---")

        return results

    def generate_report(self, results):
        """
        Generate the output report HTML
        
        Parameters :
        results : dict
            Results dictionary from run_sampler containing samples,
            acceptance rates, parameter ranges, and diagnostics.
        """
        if self.__class__.__name__ == 'MCMCBetaFixed' :
            filename = f"mcmc_results_{self.__class__.__name__}_{config.FIXED_BETA}_prior_{self.init_method}_T{round(self.maturity*365.25)}j_multivaluation_dates.html"
        else :
            filename = f"mcmc_results_{self.__class__.__name__}_prior_{self.init_method}_T{round(self.maturity*365.25)}j_multivaluation_dates.html"
        save_results_to_html(filename, self, results)

    def check_parameter_bounds(self, param_value):
        """
        Check if a parameter is within allowed bounds.
        
        Parameters :
        param_value : np.ndarray
            Parameter values to check.
        
        Returns :
        bool
            True if all parameters are within bounds, False otherwise.
        """
        min_b, max_b = self.params_bounds
        return np.all((param_value >= min_b) & (param_value <= max_b))

    def compute_log_likelihood(self, param_value):
        """
        Compute the log-likelihood.
        
        Parameters :
        param_value : np.ndarray
            SABR parameter values [alpha, beta, rho, volvol] or [alpha, rho, volvol].
        
        Returns :
        float
            Log-likelihood value.
        """
        #  ponderation temporel 
        weights = self.likelihood_weight

        maturity, forward_all_valuation_dates, market_vols_all_valuation_dates, strikes_all_valuation_dates = self.market_data_multi_valuation_dates
        
        if self.n_params == 4:
            alpha, beta, rho, volvol = param_value
        else: # n_params == 3
            alpha, rho, volvol = param_value
            beta = config.FIXED_BETA  

        likelihood = 0.0
        for i in range(len(forward_all_valuation_dates)):
            forward, market_vols, strikes = forward_all_valuation_dates[i], market_vols_all_valuation_dates[i], strikes_all_valuation_dates[i]
            sabr_vols = vol_sabr_hagan(strikes, forward, maturity, alpha, beta, rho, volvol)
            
            valid_mask = ~np.isnan(sabr_vols)
            residuals_squared = (market_vols[valid_mask] - sabr_vols[valid_mask])**2
        
            likelihood += weights[i] * np.sum(residuals_squared) 

        return - likelihood / (2 * config.OBSERVATION_ERROR**2)
        
    def compute_prior_density(self, param_value, chain_idx):
        """
        Compute the multivariate prior density.
        
        Parameters
        ----------
        param_value : np.ndarray
            Parameter values in original space.
        chain_idx : int
            Index of the chain.
        
        Returns
        -------
        float
            Prior probability density including Jacobian adjustment.
        """
        transformed_val = self.transform_parameter(param_value)
        prior_density = stats.multivariate_normal.pdf(transformed_val, self.prior_means[chain_idx], self.prior_covs[chain_idx])
        jacobian_det = self.compute_jacobian_det(param_value)
        return prior_density * jacobian_det

    def compute_prior_marginal_density(self, param_idx, param_value, chain_idx):
        """
        Compute the marginal prior density for one parameter.
        
        Parameters
        ----------
        param_idx : int
            Index of the parameter (0: alpha, 1: beta/rho, 2: rho/volvol, 3: volvol).
        param_value : float
            Value of the specific parameter.
        chain_idx : int
            Index of the chain.
        
        Returns
        -------
        float
            Marginal prior probability density.
        
        """
        # assuming diagonal covariance for our prior 
        transformed_val = self.transform_one_parameter(param_value, param_idx)
        prior_density = stats.norm.pdf(
            transformed_val, 
            self.prior_means[chain_idx, param_idx], 
            np.sqrt(self.prior_covs[chain_idx, param_idx, param_idx])
        )
        # Jacobian for this specific transformation
        jacobians = {
            0: lambda x: 1 / x, # alpha
            1: lambda x: 1 / (x * (1 - x)) if self.n_params==4 else 1 / (1 - x**2), # beta or rho
            2: lambda x: 1 / (1 - x**2) if self.n_params==4 else 1/x, # rho or volvol
            3: lambda x: 1 / x # volvol
        }
        jacobian = jacobians[param_idx](param_value)
        return prior_density * jacobian

    @abstractmethod
    def transform_one_parameter(self, value, param_index):
        """
        Transform a single parameter to unconstrained space.
        """
        pass

    def update_cov_matrix_proposal(self, k, n0, last_cov_matrix, last_mean, new_samples):
        """
        Update the proposal covariance matrix (Adaptive Metropolis).
        
        Parameters :
        k : int
            Current batch number.
        n0 : int
            Batch size.
        last_cov_matrix : np.ndarray
            Previous covariance matrix.
        last_mean : np.ndarray
            Previous mean vector.
        new_samples : np.ndarray
            New batch of samples.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Updated covariance matrix and mean vector.
        """
        d = self.n_params
        batch_mean = np.mean(new_samples, axis=0)
        new_mean = ((k - 1) / k) * last_mean + (1 / k) * batch_mean
        
        sum_outer_products = np.sum([np.outer(sample, sample) for sample in new_samples], axis=0)
        
        last_mean_col = last_mean.reshape(-1, 1)
        new_mean_col = new_mean.reshape(-1, 1)
        
        update_term = ((k - 1) * n0 * np.outer(last_mean_col, last_mean_col) 
                      - k * n0 * np.outer(new_mean_col, new_mean_col) 
                      + sum_outer_products 
                      + self.epsilon_cov * np.eye(d))
        
        new_cov_matrix = ((k - 1) / k) * last_cov_matrix + (self.s_d / (k * n0)) * update_term
        
        return new_cov_matrix, new_mean