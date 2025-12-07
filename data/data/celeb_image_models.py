import joblib
import numpy as np
import pandas as pd
from pathlib import Path

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