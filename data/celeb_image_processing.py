##Importação de bibliotecas - Precisamos ver o que de fato está sendo usado e remover o que tenha implementação pronta

import os
import time
import json
import numpy as np
import pandas as pd
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

## Apontando para local de download das imagens do dataset
IMAGES_DIR = PROJECT_ROOT / "data" / "celeba" / "img_align_celeba"   # or "img_align_celeba", etc

## Verificando presença do download das imagens
print(IMAGES_DIR.exists())     # se correto retorna TRUE

## Contagem de arquivos
print(len(os.listdir(IMAGES_DIR))) #Deve retornar 202599