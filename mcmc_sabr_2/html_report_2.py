import base64
import io
import matplotlib.pyplot as plt
import numpy as np
from mcmc_sabr.diagnostics import create_density_plot, create_trace_plot, create_autocorrelation_plot

def fig_to_base64(fig):
    """Convert matplotlib figure to base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def save_results_to_html(filename, sampler, results):
    """
    Génère un fichier HTML contenant un rapport détaillé sur les résultats MCMC.
    
    Parameters:
    filename : str
        Le nom du fichier HTML (sortie).
    sampler : object
        Objet sampler MCMC avec la méthode compute_log_likelihood et les attributs nécessaires.
    results : dict
        Dictionnaire contenant les résultats du MCMC.
    """
    samples = results['samples']
    acc_rates = results['acceptance_rates']
    
    acc_table_header = "<th>Chain</th><th>Acceptance Rate</th>"
    cov_matrix_table_html = ""

    if acc_rates.ndim == 2:  # Cas séquentiel
        acc_table_header = "<th>Chain</th>" + "".join([f"<th>{name}</th>" for name in sampler.param_names])
        acc_rows = "".join(
            f"<tr><td>{i+1}</td>" + "".join([f"<td>{rate:.2f}%</td>" for rate in acc_rates[i]]) + "</tr>"
            for i in range(acc_rates.shape[0])
        )
    else:  # Cas vectoriel
        acc_rows = "".join([f"<tr><td>{i+1}</td><td>{rate:.2f}%</td></tr>" for i, rate in enumerate(acc_rates)])
        
        correlation_matrix = np.corrcoef(samples[0].T)
        cov_table_header = "<th>Parameter</th>" + "".join([f"<th>{name}</th>" for name in sampler.param_names])
        cov_rows_list = []
        for i, param_name in enumerate(sampler.param_names):
            cells = "".join([f"<td>{val:.4f}</td>" for val in correlation_matrix[i]])
            cov_rows_list.append(f"<tr><td><b>{param_name}</b></td>{cells}</tr>")
        cov_rows = "".join(cov_rows_list)
        cov_matrix_table_html = f"""
        <div class="table-container">
            <h2>correlation Matrix (Chain 1)</h2>
            <table>
                <thead><tr>{cov_table_header}</tr></thead>
                <tbody>{cov_rows}</tbody>
            </table>
        </div>
        """

    acc_table_html = f"""
    <div class="table-container">
        <h2>Acceptance Rates</h2>
        <table><thead><tr>{acc_table_header}</tr></thead><tbody>{acc_rows}</tbody></table>
    </div>
    """

    r_hat_table_html = ""
    if sampler.diagnostics_enabled:
        r_hat_rows = "".join([f"<tr><td>{sampler.param_names[i]}</td><td>{r_val:.4f}</td></tr>" for i, r_val in enumerate(results['r_hat'])])
        r_hat_table_html = f"""
        <div class="table-container">
            <h2>Gelman-Rubin (R-hat)</h2>
            <table><thead><tr><th>Paramètre</th><th>Valeur R-hat</th></tr></thead><tbody>{r_hat_rows}</tbody></table>
        </div>
        """

    html = f"""
    <!DOCTYPE html><html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>MCMC SABR Results</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f8f9fa; color: #212529; }}
            h1, h2, h3 {{ color: #343a40; }}
            h1 {{ text-align: center; margin-bottom: 1.5em; }}
            h2 {{ border-bottom: 2px solid #dee2e6; padding-bottom: 0.5em; margin-top: 2.5em; }}
            h3 {{ margin-top: 2em; margin-bottom: 1em; }}
            .details {{ text-align: center; margin: -1em 0 2em 0; color: #495057; }}
            .details p {{ margin: 0.5em 0; font-size: 1.1em; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 30px; margin-bottom: 3em; }}
            .table-container {{ background-color: #ffffff; padding: 1.5em; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
            th {{ background-color: #f1f3f5; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .plots-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; align-items: start; margin-bottom: 2em; }}
            .plots-grid-density {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }}
            .plot {{ background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #dee2e6; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
        </style>
    </head>
    <body>
    <h1>MCMC SABR Results - Algorithm: {sampler.__class__.__name__}</h1>
    <div class="details">
        <p>Prior initialized with the <strong>{sampler.init_method}</strong> method</p>
        <p>Likelihood calculated using data of all maturities</p>
    </div>
    <div class="summary-grid">
        {r_hat_table_html}
        {acc_table_html}
        {cov_matrix_table_html}
    </div>
    """
    
    for chain_idx in range(sampler.n_chains):
        html += f"<h2>Chain {chain_idx + 1} Analysis</h2>"
        chain_samples = results['samples'][chain_idx]
        param_ranges = results['param_ranges']

        if sampler.diagnostics_enabled:
            for param_idx in range(sampler.n_params):
                param_name = sampler.param_names[param_idx]
                param_samples = chain_samples[:, param_idx]
                html += f"<h3>Parameter Analysis: {param_name}</h3>"
                html += '<div class="plots-grid">'
                density_fig = create_density_plot(sampler, param_idx, chain_idx, chain_samples, param_ranges[0][chain_idx][param_idx], param_ranges[1][chain_idx][param_idx])
                html += f'<div class="plot"><img src="data:image/png;base64,{fig_to_base64(density_fig)}" alt="Density Plot for {param_name}"></div>'
                trace_fig = create_trace_plot(param_samples, param_name)
                html += f'<div class="plot"><img src="data:image/png;base64,{fig_to_base64(trace_fig)}" alt="Trace Plot for {param_name}"></div>'
                autocorr_fig = create_autocorrelation_plot(param_samples, param_name)
                html += f'<div class="plot"><img src="data:image/png;base64,{fig_to_base64(autocorr_fig)}" alt="Autocorrelation Plot for {param_name}"></div>'
                html += '</div>'
        else: 
            html += "<h3>Parameter Posterior Densities</h3>"
            html += '<div style="display: flex; flex-direction: column; gap: 20px;">'
            for param_idx in range(sampler.n_params):
                param_samples = chain_samples[:, param_idx]
                
                density_fig = create_density_plot(
                    sampler, 
                    param_idx, 
                    chain_idx, 
                    param_samples, 
                    chain_samples,
                    param_ranges[0][chain_idx][param_idx], 
                    param_ranges[1][chain_idx][param_idx]
                )
                html += f'<div class="plot"><img src="data:image/png;base64,{fig_to_base64(density_fig)}" alt="Density Plot for {sampler.param_names[param_idx]}"></div>'
            html += '</div>'

    html += "</body></html>"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Results saved to {filename}")