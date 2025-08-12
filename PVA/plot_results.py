import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def plot_pva_surface(excel_file, file, sheet_name=0):
    """
    Plot la surface des ratios PVA en fonction de la maturité et moneyness
    
    Parameters:
    - excel_file: chemin vers le fichier Excel
    - sheet_name: nom ou index de la feuille (défaut: 0)
    """
    # Lecture du fichier Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Extraction des maturités (colonne A, lignes avec "T = ")
    maturities = []
    ratio_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).startswith('T = '):
            # Extraction de la maturité en jours
            maturity_str = str(row.iloc[0]).replace('T = ', '').replace('j', '')
            maturities.append(int(maturity_str))
            
            # Extraction des ratios (colonnes E à Q)
            ratios = []
            for col_idx in range(4, 17):  # Colonnes E à Q
                if col_idx < len(row):
                    val = row.iloc[col_idx]
                    if pd.notna(val) and val != 0:
                        ratios.append(float(val))
                    else:
                        ratios.append(np.nan)
                else:
                    ratios.append(np.nan)
            ratio_data.append(ratios)
    
    # Moneyness (strikes de 80% à 120%)
    moneyness = [80.0, 82.0, 85.0, 88.0, 90.0, 92.5, 95.0, 97.5, 
                100.0, 102.5, 105.0, 110.0, 120.0]
    
    # Conversion en arrays numpy
    maturities = np.array(maturities)
    ratio_data = np.array(ratio_data)
    moneyness = np.array(moneyness)
    
    # Création des grilles pour le plot 3D
    M, T = np.meshgrid(moneyness, maturities)
    
    # Création du plot 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(M, T, ratio_data, cmap='viridis', alpha=0.8)
    
    # Contour plot sur le plan z=0
    contours = ax.contour(M, T, ratio_data, zdir='z', offset=0, cmap='viridis', alpha=0.5)
    
    # Labels et titre
    ax.set_xlabel('Moneyness (%)')
    ax.set_ylabel('Maturité (jours)')
    ax.set_zlabel('Ratio PVA (%)')
    ax.set_title(f'Surface des Ratios PVA\n({file})')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    # Vue optimisée
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()


    
    # Plot 2D complémentaire (heatmap)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(ratio_data, cmap='viridis', aspect='auto', origin='lower')
    # plt.colorbar(label='Ratio PVA (%)')
    # plt.xlabel('Moneyness')
    # plt.ylabel('Maturité')
    # plt.title('Heatmap des Ratios PVA')
    
    # # Ticks personnalisés
    # plt.xticks(range(len(moneyness)), [f'{m}%' for m in moneyness], rotation=45)
    # plt.yticks(range(len(maturities)), [f'{t}j' for t in maturities])
    
    # plt.tight_layout()
    # plt.show()

# Utilisation
# plot_pva_surface('votre_fichier.xlsx')


def main():

    file = int(sys.argv[1])
    plot_pva_surface(f'Results_{file}/Resultats_{file}.xlsx', file)


if __name__ == "__main__":
    main()

    # print("="*40)
    # print("M", M)
    # print("T", T)
    # print("pva_data", pva_data)
    # print("="*40)