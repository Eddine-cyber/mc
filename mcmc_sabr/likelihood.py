import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
import seaborn as sns
from SABR.sabr import vol_sabr_hagan

def calculate_log_likelihood_surface(market_data, alpha_range, beta_range, fixed_beta=None):
    """
    Calcule la surface de log-vraisemblance dans le plan (α, β).
    
    Cette fonction évalue systématiquement la vraisemblance pour une grille
    de valeurs (α, β) afin de visualiser le paysage de la fonction objectif.
    """
    # Extraire les données de marché nécessaires
    maturity, forward, market_vols, strikes = market_data
    
    # Créer une grille de points pour l'évaluation
    n_points = 50  # Résolution de la grille
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    
    if fixed_beta is not None:
        # Cas avec β fixé : on explore seulement α
        betas = np.array([fixed_beta])
        log_likelihoods = np.zeros(n_points)
        
        # Pour chaque valeur d'alpha, calculer la log-vraisemblance
        for i, alpha in enumerate(alphas):
            # Fixer les autres paramètres à des valeurs raisonnables pour l'illustration
            rho = -0.7  # Valeur typique pour equity
            volvol = 1.0  # Valeur typique
            
            # Calculer les volatilités SABR pour ces paramètres
            sabr_vols = vol_sabr_hagan(strikes, forward, maturity, 
                                       alpha, fixed_beta, rho, volvol)
            
            # Calculer la log-vraisemblance (somme des carrés des écarts)
            residuals = market_vols - sabr_vols
            log_likelihoods[i] = -0.5 * np.sum(residuals**2) / (0.01**2)  # σ_err = 1%
            
        return alphas, None, log_likelihoods.reshape(-1, 1)
    
    else:
        # Cas avec β libre : on explore le plan (α, β) complet
        betas = np.linspace(beta_range[0], beta_range[1], n_points)
        log_likelihoods = np.zeros((n_points, n_points))
        
        # Double boucle pour explorer toute la grille
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Paramètres fixes pour l'illustration
                rho = -0.7
                volvol = 1.0
                
                # Calculer les volatilités SABR
                sabr_vols = vol_sabr_hagan(strikes, forward, maturity,
                                           alpha, beta, rho, volvol)
                
                # Log-vraisemblance
                residuals = market_vols - sabr_vols
                log_likelihoods[i, j] = -0.5 * np.sum(residuals**2) / (0.01**2)
        
        return alphas, betas, log_likelihoods

def plot_likelihood_comparison(market_data):
    """
    Crée une visualisation comparative des surfaces de log-vraisemblance
    avec β libre vs β fixé, révélant le problème d'identifiabilité.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. Surface avec β libre (vue du dessus - heatmap)
    alpha_range = (0.01, 2.0)
    beta_range = (0.5, 1.0)
    alphas, betas, log_lik_free = calculate_log_likelihood_surface(
        market_data, alpha_range, beta_range, fixed_beta=None
    )
    
    # Normaliser pour une meilleure visualisation
    log_lik_normalized = log_lik_free - np.max(log_lik_free)
    # log_lik_shifted = log_lik_free - np.max(log_lik_free)
    # log_lik_normalized = 1 / (1 + np.exp(-log_lik_shifted * 3))

    
    # Premier plot : Heatmap de la surface complète
    im1 = axes[0].contourf(alphas, betas, log_lik_normalized.T, 
                           levels=20, cmap='RdYlBu_r')
    axes[0].set_xlabel('α (Alpha)', fontsize=12)
    axes[0].set_ylabel('β (Beta)', fontsize=12)
    axes[0].set_title('Surface de Log-Vraisemblance\n(β libre)', fontsize=14)
    plt.colorbar(im1, ax=axes[0], label='Log-Vraisemblance (normalisée)')
    
    # Ajouter des contours pour mieux voir la forme de la "vallée"
    contours = axes[0].contour(alphas, betas, log_lik_normalized.T, 
                               levels=[-20, -10, -5, -2, -1], 
                               colors='black', alpha=0.4, linewidths=0.5)
    axes[0].clabel(contours, inline=True, fontsize=8)
    
    # 3. Vue 3D de la surface (optionnel mais très révélateur)
    from mpl_toolkits.mplot3d import Axes3D
    ax3 = fig.add_subplot(133, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    surf = ax3.plot_surface(X, Y, log_lik_normalized.T, 
                            cmap='RdYlBu_r', alpha=0.8)
    ax3.set_xlabel('α (Alpha)')
    ax3.set_ylabel('β (Beta)')
    ax3.set_zlabel('Log-Vraisemblance')
    ax3.set_title('Vue 3D de la Surface\n(β libre)', fontsize=14)
    
    # Ajouter une ligne pour montrer où β = 1
    alpha_line = alphas
    beta_line = np.ones_like(alpha_line)
    z_line = [log_lik_normalized[i, -1] for i in range(len(alphas))]
    ax3.plot(alpha_line, beta_line, z_line, 'g-', linewidth=3, 
            label='Coupe à β = 1.0')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('likelihood_surface_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Utilisation avec vos données
# plot_likelihood_comparison(votre_market_data)

def main():
    from mcmc_sabr.utils import load_market_data
    market_data = load_market_data()

    plot_likelihood_comparison(market_data)

if __name__ == "__main__":
    main()