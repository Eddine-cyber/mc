import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from SABR.prapare_data import prepare_calibration_inputs
from SABR.sabr import vol_sabr_hagan

def calculate_r_hat(chains_samples):
    """
    Calculate the Gelman-Rubin (R-hat) diagnostic for MCMC convergence assessment.
    The R-hat statistic compares the variance between chains to the variance within chains.
    Values close to 1.0 indicate good convergence (typically R-hat < 1.1 is acceptable).
    
    Parameters :
    chains_samples : np.ndarray
        A 3D array of shape (m, n, d) where:
        - m: number of independent MCMC chains
        - n: number of samples per chain
        - d: number of parameters (4 for SABR: alpha, rho, nu, beta)
    
    Returns :
    np.ndarray
        Array of shape (d,) containing R-hat values for each parameter.
        Values close to 1.0 indicate convergence.
    """

    m, n, d = chains_samples.shape
    if m <= 1:
        return np.ones(d)
    
    print("\n" + "="*60)
    print("begin Gelman-Rubin diagnostic ...")
    # Calculate within-chain variance (W): average of the variances of each chain
    chain_variances = np.var(chains_samples, axis=1, ddof=1) # ddof=1 for unbiased estimator
    W = np.mean(chain_variances, axis=0)

    # Calculate between-chain variance (B)
    # First, compute the mean of each chain
    chain_means = np.mean(chains_samples, axis=1)
    # Then, compute the overall mean across all chains
    overall_mean = np.mean(chain_means, axis=0)
    # Finally, compute the weighted variance of the chain means
    B = (n / (m - 1)) * np.sum((chain_means - overall_mean)**2, axis=0)

    # Estimate the posterior marginal variance of params (V_hat)
    V_hat = ((n - 1) / n) * W + (1 / n) * B

    # Calculate the R-hat
    R_hat = np.sqrt(V_hat / W)

    print("Gelman-Rubin diagnostic completed successfully!")
    print("\n" + "="*60)
    return R_hat


def compute_density_histogram(param_samples, param_min, param_max, interval_width=0.001):
    """
    Compute smooth density using Kernel Density Estimation.
    
    Parameters :
    param_samples : np.ndarray
        1D array of parameter samples for a specific SABR param from MCMC chain.
    param_min : float
        Minimum value for histogram range.
    param_max : float
        Maximum value for histogram range.
    interval_width : float, optional
        Width of each histogram bin (default: 0.001).
    
    Returns :
    tuple[np.ndarray, np.ndarray]
        - x_eval: array of parametre values where we want to plot
        - density: la densité à posteriori
    """
    # Créer l'estimateur KDE
    kde = gaussian_kde(param_samples, bw_method='scott') # KDE(x) = (1/n) × Σ K((x - xi)/h)
    
    # Points d'évaluation
    n_intervals = max(1, int((param_max - param_min) / interval_width))
    x_eval = np.linspace(param_min, param_max, n_intervals)
    
    # Évaluer la densité
    density = kde(x_eval)
    
    return x_eval, density

def compute_likelihood_histogram(sampler, param_samples, param_index, param_min, param_max, interval_width=0.001):
    """
    Compute likelihood profile for a specific parameter.
    
    Parameters :
    sampler : object
        MCMC sampler object with compute_log_likelihood method.
    param_samples : np.ndarray
        2D array of parameter samples for all SABR params.
    param_index : int
        Index of the parameter (0: alpha, 1: rho, 2: nu, 3: beta).
    param_min : float
        Minimum value of parameter range.
    param_max : float
        Maximum value of parameter range.
    
    Returns :
    tuple[np.ndarray, np.ndarray]
        - param_values: Array of parameter values
        - normalized_likelihoods: Normalized likelihood values at each point
    
    """
    from scipy.stats import binned_statistic
    from scipy.ndimage import gaussian_filter1d

    likelihood_values = np.exp([sampler.compute_log_likelihood(p) for p in param_samples])
    param_values = param_samples[:, param_index]
    
    bins = np.arange(param_min, param_max + interval_width, interval_width)
    
    # Average likelihoods per bin
    likelihood_means, _, _ = binned_statistic(
        param_values,
        likelihood_values,
        statistic='mean',
        bins=bins
    )
    
    # bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    valid_mask = ~np.isnan(likelihood_means)
    smoothed_likelihood = gaussian_filter1d(likelihood_means[valid_mask], sigma=10)
    integral = np.trapz(smoothed_likelihood, bin_centers[valid_mask])

    normalized_smooth_likelihoods = smoothed_likelihood / integral

    integral = np.trapz(likelihood_means[valid_mask], bin_centers[valid_mask])

    normalized_likelihoods = likelihood_means[valid_mask] / integral

    return bin_centers[valid_mask], normalized_likelihoods, normalized_smooth_likelihoods


def create_density_plot_2(sampler, param_idx, chain_idx, param_samples, param_samples_all, param_min, param_max):
    """
    Create posterior and prior density and likelihood plots.
    
    Parameters :
    sampler : object
        MCMC sampler object.
    param_index : int
        Index of the parameter (0: alpha, 1: rho, 2: nu, 3: beta).
    chain_idx : int
        Index of the chain to use.
    param_samples : np.ndarray
        1D array of parameter samples.
    param_samples_all : np.ndarray
        2D array of parameter all sabr params samples.
    param_min : float
        Minimum value of parameter range.
    param_max : float
        Maximum value of parameter range.

    Returns :
    matplotlib.figure.Figure
        Figure object .
    
    """

    global_min = min(np.min(param_min), sampler.params_bounds[0][param_idx])
    global_max = max(np.max(param_max), sampler.params_bounds[1][param_idx])

    posterior_edges, posterior_freqs = compute_density_histogram(param_samples, global_min, global_max)
    q_min, q_max = np.percentile(param_samples, [5, 95])
    posterior_mean = np.mean(param_samples)
    posterior_median = np.median(param_samples)
    
    # prior +posterior
    prior_edges = np.linspace(global_min, global_max, 500)
    transformed_edges = sampler.transform_one_parameter(prior_edges, param_idx)
    prior_densities_transformed = stats.norm.pdf(
        transformed_edges,
        sampler.prior_means[chain_idx, param_idx],
        np.sqrt(sampler.prior_covs[chain_idx, param_idx, param_idx])
    )
    jacobian_func = sampler.get_one_param_jacobian_func(param_idx)
    jacobians = jacobian_func(prior_edges)
    prior_freqs = prior_densities_transformed * jacobians

    likelihood_edges, normalized_likelihoods, normalized_smooth_likelihoods = compute_likelihood_histogram(sampler, param_samples_all, param_idx, global_min, global_max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[1,1]
    ax.plot(posterior_edges, posterior_freqs, 'b-', linewidth=2, label='Posterior Density')
    ax.plot(prior_edges, prior_freqs, 'r-', linewidth=2, label='Prior Density')
    ax.plot(likelihood_edges, normalized_smooth_likelihoods, 'g-', linewidth=2, label='likelihood')
    ax.axvline(x=sampler.initial_params[chain_idx, param_idx], color='blue', linestyle=':', label=f'Initial Guess: {sampler.initial_params[chain_idx, param_idx]:.3f}')
    ax.axvspan(q_min, q_max, alpha=0.2, color='blue', label=f'90% CI: [{q_min:.3f}, {q_max:.3f}]')
    ax.axvline(x=posterior_mean, color='blue', linestyle=':', alpha=0.7, linewidth=1.5, 
            label=f'Post. Mean: {posterior_mean:.3f}')
    prior_mean = sampler.prior_transformed_means[param_idx]
    prior_std = sampler.prior_covs[chain_idx, param_idx, param_idx]
    ax.axvline(x=posterior_median, color='blue', linestyle='-.', alpha=0.7, linewidth=1.5, 
            label=f'Post. Median: {posterior_median:.3f} \n Prior. Mean: {prior_mean:.3f}|. std: {prior_std:.3f}')
    ax.set_title(f'Posterior vs Prior for {sampler.param_names[param_idx]}')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # prior 
    ax = axes[0,0]
    ax.plot(prior_edges, prior_freqs, 'r-', linewidth=2, label='Prior Density')
    ax.fill_between(prior_edges, prior_freqs, alpha=0.3, color='red')
    ax.axvline(x=prior_mean, color='green', linestyle='-.', alpha=0.7, linewidth=1.5, 
            label=f'Prior. Mean: {prior_mean:.3f}\n Prior. std: {prior_std:.3f}')
    ax.set_title(f'Prior density for {sampler.param_names[param_idx]}')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # likelihood
    ax = axes[0,1]
    ax.plot(likelihood_edges, normalized_likelihoods, 'g-', linewidth=2, label='likelihood')
    ax.fill_between(likelihood_edges, normalized_likelihoods, alpha=0.3, color='green')
    ax.set_title(f'likelihood for {sampler.param_names[param_idx]}')
    ax.set_xlabel('x')
    ax.set_ylabel('likelihood')
    ax.grid(True, alpha=0.3)
    ax.legend()


    # posterior + prior 2
    posterior_edges, posterior_freqs = compute_density_histogram(
        param_samples, param_min, param_max
    )
    q_min, q_max = np.percentile(param_samples, [5, 95])
    posterior_mean = np.mean(param_samples)
    posterior_median = np.median(param_samples)
    prior_edges = np.linspace(param_min, param_max, 500)
    transformed_edges = sampler.transform_one_parameter(prior_edges, param_idx)
    prior_densities_transformed = stats.norm.pdf(
        transformed_edges,
        sampler.prior_means[chain_idx, param_idx],
        np.sqrt(sampler.prior_covs[chain_idx, param_idx, param_idx])
    )
    jacobian_func = sampler.get_one_param_jacobian_func(param_idx)
    jacobians = jacobian_func(prior_edges)
    prior_freqs = prior_densities_transformed * jacobians
    
    likelihood_edges, normalized_likelihoods, normalized_smooth_likelihoods = compute_likelihood_histogram(sampler, param_samples_all, param_idx, param_min, param_max)


    ax = axes[1,0]
    ax.plot(posterior_edges, posterior_freqs, 'b-', linewidth=2, label='Posterior Density')
    ax.plot(prior_edges, prior_freqs, 'r-', linewidth=2, label='Prior Density')
    ax.plot(likelihood_edges, normalized_smooth_likelihoods, 'g-', linewidth=2, label='likelihood')
    ax.axvline(x=sampler.initial_params[chain_idx, param_idx], color='blue', linestyle=':', label=f'Initial Guess: {sampler.initial_params[chain_idx, param_idx]:.3f}')
    ax.axvspan(q_min, q_max, alpha=0.2, color='blue', label=f'90% CI: [{q_min:.3f}, {q_max:.3f}]')
    ax.axvline(x=posterior_mean, color='blue', linestyle=':', alpha=0.7, linewidth=1.5, 
            label=f'Post. Mean: {posterior_mean:.3f}')
    prior_mean = sampler.prior_transformed_means[param_idx]
    prior_std = sampler.prior_covs[chain_idx, param_idx, param_idx]
    ax.axvline(x=posterior_median, color='blue', linestyle='-.', alpha=0.7, linewidth=1.5, 
            label=f'Post. Median: {posterior_median:.3f} \n Prior. Mean: {prior_mean:.3f}|. std: {prior_std:.3f}')
    ax.set_title(f'Posterior vs Prior for {sampler.param_names[param_idx]}')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend()


    plt.tight_layout()
    return fig


def create_density_plot(sampler, param_idx, chain_idx, param_samples, param_samples_all, param_min, param_max):
    """
    Create posterior and prior density and likelihood plots.
    """

    # Calcul des statistiques du posterior
    q_min, q_max = np.percentile(param_samples, [5, 95])
    posterior_mean = np.mean(param_samples)
    posterior_median = np.median(param_samples)
    
    # Info du prior
    prior_mean_transformed = sampler.prior_transformed_means[param_idx]
    prior_std = np.sqrt(sampler.prior_covs[chain_idx, param_idx, param_idx])

    
    # Calcul des bornes intelligentes pour le prior (99.9% de la masse)
    prior_lower = sampler.inverse_transform_one_parameter(
        sampler.prior_means[chain_idx, param_idx] - 4 * prior_std, 
        param_idx
    )
    prior_upper = sampler.inverse_transform_one_parameter(
        sampler.prior_means[chain_idx, param_idx] + 4 * prior_std, 
        param_idx
    )
    
    # Bornes pour vue complète (prior) et vue zoomée (posterior)
    full_min = min(param_min, prior_lower, sampler.params_bounds[0][param_idx])
    full_max = max(param_max, prior_upper, sampler.params_bounds[1][param_idx])
    
    def compute_densities(xmin, xmax):
        """Calcule toutes les densités pour un intervalle donné"""
        # Posterior
        post_edges, post_freqs = compute_density_histogram(param_samples, xmin, xmax)
        
        # Prior
        prior_edges = np.linspace(xmin, xmax, 500)
        transformed = sampler.transform_one_parameter(prior_edges, param_idx)
        prior_densities = stats.norm.pdf(
            transformed,
            sampler.prior_means[chain_idx, param_idx],
            prior_std
        )
        jacobian_func = sampler.get_one_param_jacobian_func(param_idx)
        prior_freqs = prior_densities * jacobian_func(prior_edges)
        
        # Likelihood
        lik_edges, lik_raw, lik_smooth = compute_likelihood_histogram(
            sampler, param_samples_all, param_idx, xmin, xmax
        )
        
        return {
            'posterior': (post_edges, post_freqs),
            'prior': (prior_edges, prior_freqs),
            'likelihood': (lik_edges, lik_raw, lik_smooth)
        }
    
    # Calcul unique pour chaque vue
    densities_zoom = compute_densities(param_min, param_max)
    densities_full = compute_densities(full_min, full_max)
    
    # Création de la figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Configuration commune pour les subplots principaux
    def plot_main(ax, densities, title_suffix=""):
        post_e, post_f = densities['posterior']
        prior_e, prior_f = densities['prior']
        lik_e, _, lik_smooth = densities['likelihood']
        
        ax.plot(post_e, post_f, 'b-', lw=2, label='Posterior')
        ax.plot(prior_e, prior_f, 'r-', lw=2, label='Prior')
        ax.plot(lik_e, lik_smooth, 'g-', lw=2, label='Likelihood')
        
        ax.axvline(sampler.initial_params[chain_idx, param_idx], color='blue', 
                  linestyle=':', label=f'Initial: {sampler.initial_params[chain_idx, param_idx]:.3f}')
        ax.axvspan(q_min, q_max, alpha=0.2, color='blue', 
                  label=f'90% CI: [{q_min:.3f}, {q_max:.3f}]')
        ax.axvline(posterior_mean, color='blue', linestyle=':', alpha=0.7, lw=1.5,
                  label=f'Post. Mean: {posterior_mean:.3f}')
        ax.axvline(posterior_median, color='blue', linestyle='-.', alpha=0.7, lw=1.5,
                  label=f'Post. Median: {posterior_median:.3f}\nPrior Mean: {prior_mean_transformed:.3f} | std: {prior_std:.3f}')
        
        ax.set_title(f'Posterior vs Prior for {sampler.param_names[param_idx]}{title_suffix}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Subplot principal - vue zoomée (posterior focus)
    plot_main(axes[1, 0], densities_zoom, " (zoomed)")
    
    # Subplot principal - vue complète (prior focus)
    plot_main(axes[1, 1], densities_full, " (full range)")
    
    # Prior seul
    prior_e, prior_f = densities_full['prior']
    axes[0, 0].plot(prior_e, prior_f, 'r-', lw=2, label='Prior Density')
    axes[0, 0].fill_between(prior_e, prior_f, alpha=0.3, color='red')
    axes[0, 0].axvline(prior_mean_transformed, color='green', linestyle='-.', alpha=0.7, lw=1.5,
                      label=f'Prior Mean: {prior_mean_transformed:.3f}\nPrior std: {prior_std:.3f}')
    axes[0, 0].set_title(f'Prior density for {sampler.param_names[param_idx]}')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Likelihood seul
    lik_e, lik_raw, _ = densities_full['likelihood']
    axes[0, 1].plot(lik_e, lik_raw, 'g-', lw=2, label='Likelihood')
    axes[0, 1].fill_between(lik_e, lik_raw, alpha=0.3, color='green')
    axes[0, 1].set_title(f'Likelihood for {sampler.param_names[param_idx]}')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Likelihood')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    plt.tight_layout()
    return fig


def create_trace_plot(param_samples, param_name):
    """
    Create a trace plot for MCMC parameter samples.
    
    Parameters :
    param_samples : np.ndarray
        1D array of parameter samples.
    param_name : str
        Name of the parameter to plot .
    
    Returns :
    matplotlib.figure.Figure
        Figure object .
    
    """
    print("\n" + "="*60)
    print("Beginning trace plot diagnostic...")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(param_samples, alpha=0.7, linewidth=0.8, label='Samples')
    
    ax.set_title(f'Trace Plot - {param_name}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Value of {param_name}')
    ax.grid(True, alpha=0.3)
    
    # Calculate and plot running mean to show convergence
    running_mean = np.cumsum(param_samples) / np.arange(1, len(param_samples) + 1)
    ax.plot(running_mean, 'r--', alpha=0.8, linewidth=1.5, label='Running Mean')
    
    ax.legend()
    plt.tight_layout()
    
    print("Trace plot diagnostic completed successfully!")
    print("\n" + "="*60)
    
    return fig


def create_autocorrelation_plot(param_samples, param_name):
    """
    Create an autocorrelation plot for MCMC parameter samples.
    
    Parameters :
    param_samples : np.ndarray
        1D array of parameter samples.
    param_name : str
        Name of the parameter to plot.
    
    Returns :
    matplotlib.figure.Figure
        Figure object containing the autocorrelation plot.
    
    """
    print("\n" + "="*60)
    print("Beginning autocorrelation plot diagnostic...")
    
    n_samples = len(param_samples)
    
    max_lag = min(max(n_samples // 100, 50), 500)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Calculate autocorrelation for each lag
    centered_samples = param_samples - np.mean(param_samples)
    autocorr = np.array([
        np.dot(centered_samples[:n_samples - lag], centered_samples[lag:]) / 
        np.dot(centered_samples, centered_samples)
        for lag in range(max_lag)
    ])

    lags = np.arange(max_lag)
    ax.plot(lags, autocorr, 'b-', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f'Autocorrelation - {param_name}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    print("Autocorrelation plot diagnostic completed successfully!")
    print("\n" + "="*60)
    
    return fig


def residuals_normality():
    """
    Analyze the normality of SABR model residuals across all maturities.
    
    Returns :
    np.ndarray
        Array of all residuals (market vol - SABR vol) across all strikes and maturities.
    
    """
    from datetime import datetime
    from SABR.Calibrate import  calibrate_sabr_with_beta, calibrate_sabr_all_maturities
    import mcmc_sabr.config as config
    
    # Load market data
    excel_file = '/mnt/c/Users/salaheddine.abdelhad/OneDrive - Exiom Partners/Bureau/stage-projet/sabr-mcmc-pva/Data/spx_3_apr_25.xlsx'
    valuation_date = datetime.strptime('03-Apr-2025', '%d-%b-%Y')
    # excel_file, valuation_date = config.DATA_FILE_PATH, config.INITIAL_DATE

    strikes, maturity_data = prepare_calibration_inputs(excel_file, valuation_date)
    maturities = maturity_data[1:, 0] 
    FwdImp = maturity_data[1:, 1]
    market_vols = maturity_data[1:, 2:]
    
    all_residuals = []
    
    # Perform global calibration across all maturities
    calibrated_params = calibrate_sabr_all_maturities(strikes, 
                                                      market_vols,
                                                      FwdImp, 
                                                      maturities)

    # Calculate residuals for each maturity
    for i, maturity in enumerate(maturities):
        market_vol_row = market_vols[i]
        
        # Alternative: calibrate individually for each maturity
        # calibrated_params = calibrate_sabr_with_beta(maturity, 
        #                                             strikes, 
        #                                             FwdImp[i], 
        #                                             market_vol_row)
        
        sabr_vols = vol_sabr_hagan(strikes, FwdImp[i], maturity, *calibrated_params)
        residuals = market_vol_row - sabr_vols
        all_residuals.extend(residuals)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(all_residuals), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_residuals):.4f}')
    plt.xlabel('Residuals (Market Vol - SABR Vol)')
    plt.ylabel('Density')
    plt.title('Distribution of SABR Model Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return np.array(all_residuals)


def main():
    """
    Main function to run residuals normality analysis.
    
    """
    residuals_normality()


if __name__ == "__main__":
    main()

