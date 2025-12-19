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

#Demora 30s-60s
data = joblib.load(DATASET_PATH)
X, y = data["X"], data["y"]

#Checagem das classes - somente conferência. Encontra 10177 classes
print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("n classes:", len(np.unique(y)))

# ============================
# Split: 90% train, 1% val, 9% test
# ============================
TEST_FRAC_FULL = 0.09
VAL_FRAC_FULL  = 0.01
TRAIN_FRAC_FULL = 0.90

# 1) Test: 9% do total
X_rest, X_test, y_rest, y_test = train_test_split(
    X,
    y,
    test_size=TEST_FRAC_FULL,
    random_state=42,
    shuffle=True
    # sem stratify (mantendo seu padrão)
)

# 2) Val: 1% do total, tirado dos 91% restantes
#    fração dentro do restante = 0.00 / 0.91
val_frac_rest = VAL_FRAC_FULL / (1.0 - TEST_FRAC_FULL)

X_train, X_val, y_train, y_val = train_test_split(
    X_rest,
    y_rest,
    test_size=val_frac_rest,
    random_state=42,
    shuffle=True
    # sem stratify
)

# opcional: liberar memória
del X, y, X_rest, y_rest

print("Train:", X_train.shape, y_train.shape)
print("Val:",   X_val.shape,   y_val.shape)
print("Test:",  X_test.shape,  y_test.shape)
print("Total checado:", len(y_train) + len(y_val) + len(y_test))