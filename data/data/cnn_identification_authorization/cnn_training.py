## -*- coding: utf-8 -*-
""""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Código para experimentação e treinamento de CNN
"""

# -*- coding: utf-8 -*-
"""
Preparação do dataset (CNN) a partir do output do script de ingestão RGB.

Lê:  data/cnn_identification_authorization/celeba_rgb_256x256/manifest.csv
Usa: data/cnn_identification_authorization/celeba_rgb_256x256/images/*.jpg

Gera:
- train.csv
- test.csv
no mesmo diretório do manifest.
"""

from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


# =========================
# CONFIGURAÇÕES
# =========================
# Ajustar conforme experimento
CONFIG = {
    # Data produced by  ingestion script
    "DATASET_DIR": Path(__file__).resolve().parent / "celeba_rgb_256x256",
    "MANIFEST_NAME": "manifest.csv",
    "ONLY_OK": True,                 # use ok==True if column exists

    # Class filtering
    "TOP_CLASS_FRACTION": 0.20,       # top 20% most frequent classes
    "TEST_FRACTION": 0.20,  # 20% teste
    "KFOLDS": 5,
    "SEED": 42,

    # Input
    "IMG_SIZE": 256,
    "IN_CHANNELS": 3,
    "NORM_MEAN": (0.485, 0.456, 0.406),
    "NORM_STD":  (0.229, 0.224, 0.225),

    # Augmentation (explicit, minimal)
    "AUG_HFLIP": True,
    "AUG_PAD": 4,                    # reflect-pad then random crop; 0 disables

    # Minimal ResNet (explicit)
    "RES_LAYERS": [1, 1, 1],          # blocks per stage
    "RES_CHANNELS": [64, 128, 256],   # width per stage
    "USE_BN": True,
    "ACTIVATION": "relu",
    "BLOCK_DROPOUT": 0.0,

    # Regularization
    "L2_WEIGHT": 1e-4,                # kernel L2 for Conv/Dense

    # Training
    "BATCH_SIZE": 64,
    "EPOCHS": 5,
    "LR": 1e-3,

    # Performance
    "PREFETCH": True,
    "DEVICE": "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0",

    # (Opcional) remover classes muito pequenas antes de split
    "MIN_IMAGES_PER_CLASS": 2,
}


# =========================
# FUNÇÕES
# =========================
def load_manifest(dataset_dir: Path, only_ok: bool) -> pd.DataFrame:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv não encontrado em: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = {"image_name", "label", "dst"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv sem colunas necessárias: {missing}")

    if only_ok and "ok" in df.columns:
        df = df[df["ok"] == True].copy()

    # garante tipos
    df["label"] = df["label"].astype(int)
    df["image_name"] = df["image_name"].astype(str)
    df["dst"] = df["dst"].astype(str)

    # confere existência dos arquivos (barato e evita dor depois)
    df["dst_exists"] = df["dst"].apply(lambda p: Path(p).exists())
    df = df[df["dst_exists"] == True].drop(columns=["dst_exists"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("Após filtros, não sobrou nenhuma imagem válida.")

    return df


def keep_top_classes(df: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    if not (0 < top_fraction <= 1.0):
        raise ValueError("TOP_CLASS_FRACTION deve estar em (0, 1].")

    if top_fraction >= 1.0:
        return df

    counts = df["label"].value_counts()
    n_classes = len(counts)
    k = max(1, int(math.ceil(top_fraction * n_classes)))
    top_labels = set(counts.nlargest(k).index.tolist())

    df2 = df[df["label"].isin(top_labels)].reset_index(drop=True)
    print(f"[INFO] Mantendo top classes: {k}/{n_classes} ({100*k/n_classes:.1f}%)")
    print(f"[INFO] Imagens após filtro top-classes: {len(df2)}/{len(df)} ({100*len(df2)/len(df):.1f}%)")
    return df2


def drop_small_classes(df: pd.DataFrame, min_per_class: int) -> pd.DataFrame:
    if min_per_class <= 1:
        return df

    counts = df["label"].value_counts()
    keep = set(counts[counts >= min_per_class].index.tolist())
    df2 = df[df["label"].isin(keep)].reset_index(drop=True)

    removed_classes = (counts < min_per_class).sum()
    if removed_classes > 0:
        print(f"[INFO] Removidas {removed_classes} classes com < {min_per_class} imagens.")
        print(f"[INFO] Imagens após filtro min_per_class: {len(df2)}/{len(df)} ({100*len(df2)/len(df):.1f}%)")

    if df2.empty:
        raise RuntimeError("Após remover classes pequenas, dataset ficou vazio.")
    return df2


def stratified_train_test_split(df: pd.DataFrame, test_fraction: float, seed: int):
    """
    Split estratificado por classe sem sklearn.
    """
    if not (0 < test_fraction < 1.0):
        raise ValueError("TEST_FRACTION deve estar em (0,1).")

    # embaralha com seed
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_parts = []
    train_parts = []

    for label, grp in df.groupby("label", sort=False):
        n = len(grp)
        n_test = int(round(n * test_fraction))
        # garante pelo menos 1 no treino e 1 no teste quando possível
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
        else:
            # n==1: força tudo no treino (ou você remove antes com MIN_IMAGES_PER_CLASS=2)
            n_test = 0

        test_parts.append(grp.iloc[:n_test])
        train_parts.append(grp.iloc[n_test:])

    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = pd.concat(train_parts, ignore_index=True)

    # embaralha de novo só por estética/consistência
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df


def save_splits(dataset_dir: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    cols = ["image_name", "label", "dst"]
    train_df[cols].to_csv(train_path, index=False)
    test_df[cols].to_csv(test_path, index=False)

    print("[DONE] Splits salvos:")
    print(" -", train_path)
    print(" -", test_path)
    print(f"[DONE] Train: {len(train_df)} | Test: {len(test_df)} | Total: {len(train_df)+len(test_df)}")

    print("[INFO] #classes (train):", train_df["label"].nunique())
    print("[INFO] #classes (test): ", test_df["label"].nunique())



# =========================
# MAIN
# =========================
def main():
    dataset_dir: Path = CONFIG["DATASET_DIR"]
    only_ok: bool = CONFIG["ONLY_OK"]
    top_fraction: float = CONFIG["TOP_CLASS_FRACTION"]
    test_fraction: float = CONFIG["TEST_FRACTION"]
    seed: int = CONFIG["SEED"]
    min_per_class: int = CONFIG["MIN_IMAGES_PER_CLASS"]

    print("[INFO] Dataset dir:", dataset_dir)
    df = load_manifest(dataset_dir, only_ok=only_ok)
    print("[INFO] Manifest carregado:", df.shape)

    df = keep_top_classes(df, top_fraction=top_fraction)
    df = drop_small_classes(df, min_per_class=min_per_class)

    train_df, test_df = stratified_train_test_split(df, test_fraction=test_fraction, seed=seed)
    save_splits(dataset_dir, train_df, test_df)


if __name__ == "__main__":
    main()
