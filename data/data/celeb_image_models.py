import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

#Antes de rodar lembrar de rodar o HOG e definir qual a % dos dados!
#Para experimentação usar HOG com 30% dos dados

# Configurando diretórios
DATA_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / "celeba_hog_128x128_o9.joblib"

print("Dataset path:", DATASET_PATH)
print("Exists?", DATASET_PATH.exists())

#Demora 30s para 30% dos dados, 2 min para tudo
data = joblib.load(DATASET_PATH)
X, y = data["X"], data["y"]

#Checagem das classes - somente conferência. Encontra 10177 classes
print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("n classes:", len(np.unique(y)))

# ============================
# 1) Test: 15% do total
# ============================
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True      # padrão, mas deixo explícito
    # sem stratify
)

# ============================
# 2) Intro (5%) e Holdout (80%) dentro dos 85% restantes
# ============================
INTRO_FRAC_FULL = 0.05
HOLDOUT_FRAC_FULL = 0.80
REST_FRAC = INTRO_FRAC_FULL + HOLDOUT_FRAC_FULL  # 0.85

holdout_frac_rest = HOLDOUT_FRAC_FULL / REST_FRAC  # 80/85 ≈ 0.94117

X_intro, X_holdout, y_intro, y_holdout = train_test_split(
    X_temp,
    y_temp,
    test_size=holdout_frac_rest,
    random_state=42,
    shuffle=True      # também sem stratify aqui
)

# opcional: liberar memória
del X, y, X_temp, y_temp

print("Intro:",   X_intro.shape,  y_intro.shape)
print("Holdout:", X_holdout.shape, y_holdout.shape)
print("Test:",    X_test.shape,   y_test.shape)
print("Total checado:", len(y_intro) + len(y_holdout) + len(y_test))