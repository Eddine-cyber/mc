from SABR.sabr import calculate_vanilla_price, vol_sabr_hagan_one_strike
import numpy as np
import matplotlib.pyplot as plt
import mcmc_sabr.config as config
import mcmc_sabr.utils as utils



def calculate_pva(sampler, option, samples, f, maturity, k, implied_vol, interest_rate):
    prices = np.empty(len(samples))

    for i in range(len(samples)):
        alpha, beta, rho, volvol = samples[i]
        sigma = vol_sabr_hagan_one_strike(k, f, maturity, alpha, beta, rho, volvol)
        prices[i] = calculate_vanilla_price(option, sigma, f, k, maturity, interest_rate)

    #calcule du pris reference 
    ref_price = calculate_vanilla_price(option, implied_vol, f, k, maturity, interest_rate)


    sorted_indices = np.argsort(prices)
    sorted_prices = prices[sorted_indices]
    sorted_samples = samples[sorted_indices]
    percentile_index = int(len(samples) * 0.10) # Index pour 10%
    percentile_10 = sorted_prices[percentile_index]

    pva = ref_price - percentile_10

    weights, cumsum_weights = np.zeros(len(samples)), np.zeros(len(samples))
    return pva, ref_price, percentile_10, prices, weights, sorted_prices, cumsum_weights


def calculate_pva_weighted(sampler, option, samples, f, maturity, k, implied_vol, interest_rate):

    n_samples = len(samples)
    prices = np.empty(n_samples)
    for i in range(n_samples):
        if sampler.n_params == 4:
            alpha, beta, rho, volvol = samples[i]
        else :
            alpha, rho, volvol = samples[i]
            beta = 1.0
        sigma = vol_sabr_hagan_one_strike(k, f, maturity, alpha, beta, rho, volvol)
        prices[i] = calculate_vanilla_price(option, sigma, f, k, maturity, interest_rate)

    # Calculer la densité de chaque échantillon
    weights = utils.calculate_posterior_density(sampler, samples)
    
    # Quantile pondéré
    sorted_idx = np.argsort(prices)
    sorted_prices = prices[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    cumsum_weights = np.cumsum(sorted_weights)
    percentile_10 = sorted_prices[cumsum_weights >= 0.1][0]
    # alpha, beta, rho, volvol = sampler.initial_params[0]
    # calcule du prix de reference 
    ref_price = calculate_vanilla_price(option, implied_vol, f, k, maturity, interest_rate)
    
    return ref_price - percentile_10, ref_price, percentile_10, prices, weights, sorted_prices, cumsum_weights

# def calculate_pva_weighted_vol(sampler, option, samples, f, maturity, k, implied_vol, interest_rate):

#     n_samples = len(samples)
#     prices, vols = np.zeros(n_samples), np.zeros(n_samples)
#     for i in range(n_samples):
#         if sampler.n_params == 4:
#             alpha, beta, rho, volvol = samples[i]
#         else :
#             alpha, rho, volvol = samples[i]
#             beta = 1.0
#         sigma = vol_sabr_hagan_one_strike(k, f, maturity, alpha, beta, rho, volvol)
#         vols[i] = sigma
#         prices[i] = calculate_vanilla_price(option, sigma, f, k, maturity, interest_rate)

    
#     # Quantile pondéré
#     sorted_idx = np.argsort(prices)
#     sorted_prices = prices[sorted_idx]
#     sorted_vols = vols[sorted_idx]
    
#     percentile_index = int(n_samples * 0.10)
#     vols_percentil = np.cumsum(sorted_idx)
#     percentile_10 = sorted_prices[cumsum_weights >= 0.1][0]
#     # alpha, beta, rho, volvol = sampler.initial_params[0]
#     # calcule du prix de reference 
#     ref_price = calculate_vanilla_price(option, implied_vol, f, k, maturity, interest_rate)
    
#     return ref_price - percentile_10, ref_price, percentile_10, prices, weights, sorted_prices, cumsum_weights

def smiles_distribution(sampler, samples, f, maturity, strikes):
    
    n_strikes = len(strikes)
    vols_percentil_10, vols_percentil_90, vols_mean = np.zeros(n_strikes), np.zeros(n_strikes), np.zeros(n_strikes)
    for i in range(13):#[0,2,4,6,8,10,12]:
        k = strikes[i]

        n_samples = len(samples)
        prices, vols = np.zeros(n_samples), np.zeros(n_samples)
        for j in range(n_samples):
            if sampler.n_params == 4:
                alpha, beta, rho, volvol = samples[j]
            else :
                alpha, rho, volvol = samples[j]
                beta = config.FIXED_BETA
            sigma = vol_sabr_hagan_one_strike(k, f, maturity, alpha, beta, rho, volvol)
            vols[j] = sigma

        vols_percentil_10[i] = np.percentile(vols, 10)
        vols_mean[i]  = np.mean(vols)
        vols_percentil_90[i] = np.percentile(vols, 90)

    return vols_percentil_10, vols_mean, vols_percentil_90

def plot_smiles_distribution(sampler, samples, f, maturity, strikes, strike_percentages, market_vols, Likelihood_methode):

    vols_percentil_10, vols_mean, vols_percentil_90 = smiles_distribution(sampler, samples, f, maturity, strikes)
    
    plt.figure(figsize=(11, 6))
    plt.title(f'T={round(maturity*365.25)} J | Likelihood_methode : {Likelihood_methode}')
    plt.plot(strike_percentages, market_vols, 'o', color='b', alpha=0.7, label=f'implied vols')
    plt.plot(strike_percentages, vols_percentil_90, 'o', color='g', alpha=0.7, label=f'vol\'s 90th percentil')
    plt.plot(strike_percentages, vols_mean, 'o', color='r', alpha=0.7, label='vol\'s mean') 
    plt.plot(strike_percentages, vols_percentil_10, 'o', color='y', alpha=0.7, label='vol\'s 10th percentil')
    plt.legend()
    plt.xlabel('Strike (%)')
    plt.ylabel('Volatility') 
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results_smile_calibration_with_mcmc, T={round(maturity*365.25)} J | Likelihood_methode : {Likelihood_methode}", transparent=True)
    

def plot_pva_results(sampler, option, samples, f, maturity, k, implied_vol, s_0, interest_rate):
    """
    Compare les deux méthodes de calcul PVA et crée des visualisations
    """

    pva, ref_price, percentile_10, prices, weights, sorted_prices, cumsum_weights = calculate_pva_weighted(sampler, option, samples, f, maturity, k, implied_vol, interest_rate)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogramme des prix
    ax1 = axes[0, 0]
    ax1.hist(prices, bins=200, alpha=0.7, density=True, color='skyblue', label='Distribution des prix')
    ax1.axvline(ref_price, color='green', linestyle='--', linewidth=2,
                label=f'Prix ref : {ref_price}')
    ax1.axvline(percentile_10, color='green', linestyle=':', linewidth=2,
                label=f'10e percentile : {percentile_10}')
    ax1.set_xlabel('Prix de l\'option')
    ax1.set_ylabel('Densité')
    ax1.set_title('Distribution des prix d\'option')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # plot prix vs densité
    ax2 = axes[0, 1]
    scatter = ax2.scatter(prices, weights, alpha=0.5, c=weights, cmap='viridis')
    ax2.set_xlabel('Prix de l\'option')
    ax2.set_ylabel('Densité postérieure')
    ax2.set_title('Prix vs Densité postérieure')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Densité')
    
    # Fonction de répartition
    ax3 = axes[1, 0]
    # CDF empirique
    # y_empirical = np.arange(1, len(prices) + 1) / len(prices)
    # ax3.plot(sorted_prices, y_empirical, 'b-', alpha=0.7, label='CDF empirique')
    
    # CDF pondérée
    ax3.plot(sorted_prices, cumsum_weights, 'r-', alpha=0.7, label='CDF')
    
    # Marquer les percentiles
    ax3.axhline(0.1, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(percentile_10, color='red', linestyle=':', alpha=0.7)
    
    ax3.set_xlabel('Prix de l\'option')
    ax3.set_ylabel('CDF')
    ax3.set_title('Fonctions de répartition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparaison des PVA
    ax4 = axes[1, 1]
    pva_values = np.array([pva])
    ref_prices = np.array([ref_price])
    percentiles = np.array([percentile_10])
    
    width = 0.25
    
    bars1 = ax4.bar(- width, 1, width, label='Prix de référence', alpha=0.7)
    bars2 = ax4.bar(0, percentiles/ref_prices, width, label='10e percentile', alpha=0.7)
    bars3 = ax4.bar(width, pva_values/ref_prices, width, label='PVA', alpha=0.7)
    
    ax4.set_ylabel('Valeur')
    ax4.set_title('Résultats')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars3:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    strike_percentage = round(k/s_0*100)
    plt.suptitle(f"[{option} k={strike_percentage}% T={round(maturity*365.25)} Froward_at_maturity={f}]", 
                 fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.94)

    
    return fig, {
        'pva': pva,
        'ref_price': ref_price,
        'prices': prices,
        'weights': weights
    }

