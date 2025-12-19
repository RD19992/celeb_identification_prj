##Importação de bibliotecas - Precisamos ver o que de fato está sendo usado e remover o que tenha implementação pronta

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from joblib import Parallel, delayed
from skimage import io
from skimage.feature import hog
from joblib import dump
import math

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

#Seleção por classe

TOP_CLASS_FRACTION = 0.20  # 20% das classes com mais imagens

class_counts = df["label"].value_counts()  # label -> count
n_classes = class_counts.shape[0]
k = max(1, int(math.ceil(TOP_CLASS_FRACTION * n_classes)))

top_classes = class_counts.nlargest(k).index.astype(np.int64)
top_classes_set = set(top_classes.tolist())

df_top = df[df["label"].isin(top_classes_set)].reset_index(drop=True)

print(f"Top classes: {k}/{n_classes} ({100*k/n_classes:.1f}%)")
print(f"Images kept after class filter: {len(df_top)}/{len(df)} ({100*len(df_top)/len(df):.1f}%)")


# Checagem visual das imagens

print("Sample:", list(identities.items())[:5])

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


# Para reduzir tempos de processamento vamos reduzir tamanho para 64 x 64
IMG_SIZE = (64, 64)

# O HOG usa somente intensidades. Vamos usar tons de cinza
def preprocess_image_cv2(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)      # uint8 [0,255]
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)

# Amostra aleatória de 5 imagens do dataframe processado
sample = df.sample(5, random_state=42)

plt.figure(figsize=(12, 4))

for i, row in enumerate(sample.itertuples(), start=1):
    # usa o mesmo diretório das imagens originais
    img_path = os.path.join(IMAGES_DIR, row.image_name)  # troque IMAGES_DIR se o seu nome for outro

    # aplica pré-processamento para amostra de 5 imagens (128x128, escala de cinza, [0,1])
    img = preprocess_image_cv2(img_path)

    plt.subplot(1, 5, i)
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Label: {row.label}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# =================================================
# =================================================
# Aplicando o metodo HOG, com execução paralela
# =================================================
# =================================================

# =========================
# Configurações ajustáveis para o HOG
# =========================

# porcentagem das imagens a processar (1.0 = 100%, 0.1 = 10%, etc.)
PERCENT_IMAGES = 1.00 #Fazendo HOG para 100% das imagens após seleção de classes

# número de processos paralelos (-1 = todos os núcleos)
N_JOBS = -1

# caminho de saída do dataset HOG
OUTPUT_PATH = "data/celeba_hog_128x128_o9.joblib"

# =========================
# FUNÇÃO HOG
# =========================

def extract_hog(img, orientations, pixels_per_cell, cells_per_block):
    """Extrai o vetor HOG de uma imagem já em escala de cinza [0,1]."""
    feat = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat

# =========================
# FUNÇÃO AUXILIAR
# =========================

def hog_from_path(path):
    """Lê a imagem, aplica pré-processamento e extrai HOG."""
    img = preprocess_image_cv2(path)  # 128x128, cinza, [0,1]
    feat = extract_hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    )
    return feat.astype(np.float32)

# =========================
# CONSTRUÇÃO DO DATASET
# =========================

if not (0 < PERCENT_IMAGES <= 1.0):
    raise ValueError("PERCENT_IMAGES deve estar entre 0 e 1.")

n_total = len(df_top)
n_use = int(np.ceil(n_total * PERCENT_IMAGES))

df_subset = df_top.sample(n=n_use, random_state=42).reset_index(drop=True)

print(f"Usando {n_use} de {n_total} imagens (apenas top {TOP_CLASS_FRACTION*100:.0f}% classes). "
      f"({PERCENT_IMAGES * 100:.1f}%).")

# caminhos das imagens e labels correspondentes
paths = [os.path.join(IMAGES_DIR, name) for name in df_subset["image_name"]]
y = df_subset["label"].values.astype(np.int64)

# garante que a pasta de saída exista
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# extração de HOG em paralelo
features = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(hog_from_path)(p) for p in paths
)

X = np.vstack(features)

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print("Tamanho aproximado em RAM (GB):", X.nbytes / 1024**3)

# salvar com joblib
dump({"X": X, "y": y}, OUTPUT_PATH, compress=3)
print(f"Dataset HOG salvo em: {OUTPUT_PATH}")
