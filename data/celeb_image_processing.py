##Importação de bibliotecas - Precisamos ver o que de fato está sendo usado e remover o que tenha implementação pronta

import os
import time
import json
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage import io, color, transform
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump

## Configurando diretórios

## Definindo local atual
HERE = Path(__file__).resolve().parent

## Definindo raiz do local atual
PROJECT_ROOT = HERE.parent

## Apontando para local de download das imagens do dataset e arquivo de identificação de pessoas
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGES_DIR = DATA_DIR / "img_align_celeba"
IDENTITY_FILE = DATA_DIR / "identity_CelebA.txt"

## Verificando presença do download das imagens
print(IMAGES_DIR.exists())     # se correto retorna TRUE

## Contagem de arquivos
print(len(os.listdir(IMAGES_DIR))) #Deve retornar 202599

## Verificando se existe o mapeamento de imagem e pessoa (número de identificação)

def load_identities(identity_path: Path = IDENTITY_FILE):
    mapping = {}
    with identity_path.open() as f:
        for line in f:
            fname, ident = line.strip().split()
            mapping[fname] = int(ident)
    return mapping

identities = load_identities()
print("Num identities:", len(identities))
print("Sample:", list(identities.items())[:5])

# Criação de labels

df = pd.read_csv(
    IDENTITY_FILE,        # usa o Path com nome de variável
    sep=" ",              # separador em branco
    header=None
)
df.columns = ["image_name", "label"]

labels_path = DATA_DIR / "labels.csv"
df.to_csv(labels_path, index=False)

print("Labels salvos em:", labels_path)
print(df.head())

# Checagem visual das imagens

sample = df.sample(5)
plt.figure(figsize=(10,5))
for i, row in enumerate(sample.itertuples()):
    img_path = os.path.join(IMAGES_DIR, row.image_name)
    img = io.imread(img_path)

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Label: {row.label}")

plt.tight_layout()
plt.show()
