import numpy as np
from datetime import datetime

# 'sequential', 'vectorized', 'mala', 'beta_fixed'
DEFAULT_ALGO = 'vectorized'
N_ITERATIONS = 100000
N_CHAINS = 1
ADAPTIVE_COVARIANCE = True
DIAGNOSTICS_ENABLED = True
INITIALIZATION_METHOD = 'calibrate'  # 'calibrate' or 'manual'
PARAM_NAMES = ["alpha", "beta", "rho", "volvol"]
FIXED_BETA = 0.999

# ================== MODELE Params ==================
MATURITY_INDEX = 4
OBSERVATION_ERROR = 0.1
DATA_FILE_PATH = 'DATA/mid.xlsx'
INITIAL_DATE = datetime.strptime('31-Oct-2022', '%d-%b-%Y')

# =======================================================
dates_aug = ['11', '07', '04', '01']
dates_jul = ['29', '25', '23']

DATA_FILES_PATH = []
INITIAL_DATES = []

for day, month in [(d, '08') for d in dates_aug] + [(d, '07') for d in dates_jul]:
    DATA_FILES_PATH.append(f'DATA/likelihood_data/{day}_{month}_2025.xlsx')
    INITIAL_DATES.append(datetime.strptime(f'{day}_{month}_2025', '%d_%m_%Y'))

# =======================================================

# Scaling parameter for adaptive covariance
S_D_4_PARAMS = (2.4**2) / 4  # Pour 4 paramètres
S_D_3_PARAMS = (2.4**2) / 3  # Pour 3 paramètres

# Paramètres MALA
LANGEVIN_ERROR = 0.2
FINITE_DIFF_H = 1e-7

BOUNDS_ERROR = 1e-4

# ================== Bornes des params ==================

# Bornes pour les 4 paramètres [alpha, beta, rho, volvol]
FOUR_PARAM_BOUNDS = (
    np.array([BOUNDS_ERROR, BOUNDS_ERROR, -1.0 + BOUNDS_ERROR, BOUNDS_ERROR]),
    np.array([3.0, 1.0 - BOUNDS_ERROR, 1.0 - BOUNDS_ERROR, 5.0])
)

# Bornes pour les 3 paramètres [alpha, rho, volvol] avec beta fixe
THREE_PARAM_BOUNDS = (
    np.array([BOUNDS_ERROR, -1.0 + BOUNDS_ERROR, BOUNDS_ERROR]),
    np.array([10.0, 1.0 - BOUNDS_ERROR, 10.0])
)

# ================== config DU PRIORS ==================

PRIOR_CONFIDENCE_INTERVAL = 99

# PRIOR_BOUNDS_4_PARAMS = np.array([
#     [0.1, 1.0],     # alpha 
#     [0.4, 0.99],    # beta
#     [-0.95, 0.2],    # rho
#     [0.1, 2.5]      # volvol
# ])

PRIOR_BOUNDS_4_PARAMS = np.array([
    [0.1, 1.0],     # alpha 
    [0.2, 0.8],    # beta
    [-0.95, 0.2],    # rho
    [0.1, 2.5]      # volvol
])

PRIOR_BOUNDS_3_PARAMS = np.array([
    [0.1, 2.0],     # alpha
    [-0.99, 0.3],    # rho
    [0.1, 3.0]      # volvol
])

# PRIOR_BOUNDS_3_PARAMS = np.array([
#     [0.005, 5.0],     # alpha
#     [-0.999, 0.999],    # rho
#     [0.005, 10.0]      # volvol
# ])

# ================== PARAMETRES INITIAUX (si 'manual') ==================
MANUAL_INITIAL_PARAMS_4 = np.array([
    [0.5, 0.5, -0.3,  0.5],
    [0.35, 0.75, -0.55, 2.2],
    [0.15, 0.85, -0.75, 1.5],
    [0.45, 0.90, -0.45, 2.5]
])

MANUAL_INITIAL_PARAMS_3 = np.array([
    [0.5, -0.3,  0.5], # correspondent à beta = 0.999
    [0.35, -0.55, 2.2],
    [0.15, -0.75, 1.5],
    [0.45, -0.45, 2.5]
])

MANUAL2_INITIAL_PARAMS_4 = np.array([
    [0.1, 0.8, -0.6,  0.4],
    [0.35, 0.75, -0.55, 2.2],
    [0.15, 0.85, -0.75, 1.5],
    [0.45, 0.90, -0.45, 2.5]
])

MANUAL2_INITIAL_PARAMS_3 = np.array([
    [0.1, -0.6,  0.4], # correspondent à beta = 0.999
    [0.35, -0.55, 2.2],
    [0.15, -0.75, 1.5],
    [0.45, -0.45, 2.5]
])
