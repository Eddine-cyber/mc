import numpy as np
import pandas as pd

# ===================================== Fonctions de préparation des données =====================================

def prepare_calibration_inputs(excel_file, first_date):
    """
    Prépare les données d'entrée pour la calibration à partir d'un fichier Excel
    
    Args:
        excel_file: Chemin vers le fichier Excel contenant les données
        valuation_date: Date d'évaluation
        
    Returns:
        Tuple (strike_percentages, mat_data)
    """
    df = pd.read_excel(excel_file, decimal=',') # 'Date exp', 'FwdImp', '95.0%', '96.0%' ... '109.0%', '110.0%'
    strikes = df.iloc[0, 2:].values.astype(float)
    market_data_df = df.iloc[1:].copy()

    # mat_data : T, Fwd, puis toutes les colonnes de Vol

    # Convertir les dates d'expiration en maturités (années)
    market_data_df['T'] = (pd.to_datetime(df['Date exp']) - pd.to_datetime(first_date)).dt.days / 365.25
    
    # Sélectionner les colonnes pertinentes
    vol_columns = df.columns[2:]
    cols_for_matrix = ['T', 'FwdImp'] + list(vol_columns)
    mat_data = market_data_df[cols_for_matrix].values.astype(float) # 'T', 'FwdImp', '95.0%', '96.0%' ... '109.0%', '110.0%'
    
    # Convertir les volatilités de % en décimal
    mat_data[:, 2:] = mat_data[:, 2:] / 100.0

    return strikes, mat_data

def prepare_calibration_inputs_K_T(excel_file, first_date):

    df = pd.read_excel(excel_file, decimal=',') # 'Date exp', 'FwdImp', '95.0%', '96.0%' ... '109.0%', '110.0%'
    prcnt = df.columns[2:].to_list()
    moneyness = np.array([float(p.strip('%'))/100 for p in prcnt])

    strikes = df.iloc[0, 2:].values.astype(float)
    market_data_df = df.iloc[1:].copy()

    # mat_data : T, Fwd, puis toutes les colonnes de Vol

    # Convertir les dates d'expiration en maturités (années)
    market_data_df['T'] = (pd.to_datetime(df['Date exp']) - pd.to_datetime(first_date)).dt.days / 365.25
    
    # Sélectionner les colonnes pertinentes
    vol_columns = df.columns[2:]
    cols_for_matrix = ['T', 'FwdImp'] + list(vol_columns)
    mat_data = market_data_df[cols_for_matrix].values.astype(float) # 'T', 'FwdImp', '95.0%', '96.0%' ... '109.0%', '110.0%'
    
    # Convertir les volatilités de % en décimal
    mat_data[:, 2:] = mat_data[:, 2:] / 100.0

    return moneyness, strikes, mat_data
