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
# FUNÇÕES BÁSICAS
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
# FUNÇÕES BLOCO CNN RESNET
# =========================

def conv_bn_act(
    x,
    filters: int,
    k: int,
    stride: int,
    use_bn: bool,
    activation: str,
    l2_weight: float,
    name: str
):
    x = layers.Conv2D(
        filters, k, strides=stride, padding="same", use_bias=not use_bn,
        kernel_regularizer=regularizers.l2(l2_weight),
        name=f"{name}_conv{k}x{k}"
    )(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_{activation}")(x)
    return x


def basic_res_block(
    x,
    filters: int,
    stride: int,
    use_bn: bool,
    activation: str,
    dropout_p: float,
    l2_weight: float,
    name: str
):
    """
    BasicBlock:
      (3x3 -> BN -> Act) -> (3x3 -> BN) + skip -> Act
    Downsample skip with 1x1 when stride!=1 or channels change.
    """
    shortcut = x

    in_ch = x.shape[-1]
    out_ch = filters

    # main path
    y = conv_bn_act(
        x, filters=filters, k=3, stride=stride,
        use_bn=use_bn, activation=activation, l2_weight=l2_weight,
        name=f"{name}_c1"
    )
    y = conv_bn_act(
        y, filters=filters, k=3, stride=1,
        use_bn=use_bn, activation=None, l2_weight=l2_weight,
        name=f"{name}_c2"
    )

    if dropout_p and dropout_p > 0:
        y = layers.SpatialDropout2D(dropout_p, name=f"{name}_drop")(y)

    # skip path adjustment if needed
    if stride != 1 or (in_ch is not None and int(in_ch) != int(out_ch)):
        shortcut = layers.Conv2D(
            out_ch, 1, strides=stride, padding="same", use_bias=not use_bn,
            kernel_regularizer=regularizers.l2(l2_weight),
            name=f"{name}_skip_conv1x1"
        )(shortcut)
        if use_bn:
            shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    # merge + activation
    out = layers.Add(name=f"{name}_add")([shortcut, y])
    out = layers.Activation(activation, name=f"{name}_out_{activation}")(out)
    return out


def make_stage(
    x,
    filters: int,
    n_blocks: int,
    first_stride: int,
    use_bn: bool,
    activation: str,
    dropout_p: float,
    l2_weight: float,
    name: str
):
    # first block can downsample
    x = basic_res_block(
        x, filters=filters, stride=first_stride,
        use_bn=use_bn, activation=activation,
        dropout_p=dropout_p, l2_weight=l2_weight,
        name=f"{name}_b1"
    )
    for i in range(2, n_blocks + 1):
        x = basic_res_block(
            x, filters=filters, stride=1,
            use_bn=use_bn, activation=activation,
            dropout_p=dropout_p, l2_weight=l2_weight,
            name=f"{name}_b{i}"
        )
    return x


def build_min_resnet(cfg, num_classes: int) -> Model:
    """
    Minimal ResNet (explicit):
    stem -> stages -> GAP -> Dense
    """
    inp = layers.Input(shape=(cfg["IMG_SIZE"], cfg["IMG_SIZE"], cfg["IN_CHANNELS"]), name="input")

    # Stem (simple 3x3)
    x = conv_bn_act(
        inp, filters=cfg["RES_CHANNELS"][0], k=3, stride=1,
        use_bn=cfg["USE_BN"], activation=cfg["ACTIVATION"],
        l2_weight=cfg["L2_WEIGHT"], name="stem"
    )

    # Stages
    layers_per_stage = cfg["RES_LAYERS"]
    chs = cfg["RES_CHANNELS"]
    for s, (n_blocks, filters) in enumerate(zip(layers_per_stage, chs), start=1):
        stride = 1 if s == 1 else 2
        x = make_stage(
            x, filters=filters, n_blocks=n_blocks, first_stride=stride,
            use_bn=cfg["USE_BN"], activation=cfg["ACTIVATION"],
            dropout_p=cfg["BLOCK_DROPOUT"], l2_weight=cfg["L2_WEIGHT"],
            name=f"stage{s}"
        )

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    logits = layers.Dense(
        num_classes,
        kernel_regularizer=regularizers.l2(cfg["L2_WEIGHT"]),
        name="logits"
    )(x)

    return Model(inputs=inp, outputs=logits, name="ResNetSmallExplicit")

# =========================
# DATA LOADING + STRATIFIED KFOLD (NO SKLEARN)
# =========================
import math
import random
import numpy as np
import pandas as pd

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_and_filter_manifest(cfg):
    manifest_path = cfg["DATASET_DIR"] / cfg["MANIFEST_NAME"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    need = {"dst", "label"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"manifest must contain columns {need}, got {df.columns.tolist()}")

    if cfg["ONLY_OK"] and "ok" in df.columns:
        df = df[df["ok"] == True].copy()

    df["dst"] = df["dst"].astype(str)
    df["label"] = df["label"].astype(int)

    # keep only existing files
    df = df[df["dst"].apply(lambda p: Path(p).exists())].copy()
    df.reset_index(drop=True, inplace=True)

    # top classes by frequency
    top_fraction = cfg["TOP_CLASS_FRACTION"]
    if not (0 < top_fraction <= 1.0):
        raise ValueError("TOP_CLASS_FRACTION must be in (0,1].")

    if top_fraction < 1.0:
        counts = df["label"].value_counts()
        k = max(1, int(math.ceil(top_fraction * len(counts))))
        keep_labels = set(counts.nlargest(k).index.tolist())
        df = df[df["label"].isin(keep_labels)].copy()
        df.reset_index(drop=True, inplace=True)

    # remap labels to [0..C-1]
    uniq = sorted(df["label"].unique().tolist())
    label2idx = {lab: i for i, lab in enumerate(uniq)}
    df["y"] = df["label"].map(label2idx).astype(int)

    # ensure each class has enough samples for k-fold
    kfolds = cfg["KFOLDS"]
    counts = df["y"].value_counts()
    keep_y = set(counts[counts >= kfolds].index.tolist())
    df = df[df["y"].isin(keep_y)].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise RuntimeError("Dataset empty after class filtering for k-fold.")

    print(f"[INFO] images={len(df)} classes={df['y'].nunique()}")
    return df, label2idx


def stratified_kfold_indices(df: pd.DataFrame, k: int, seed: int):
    """
    For each class y: shuffle indices, split into k chunks.
    Fold i uses chunk i from every class as validation.
    """
    per_y = {}
    for y, grp in df.groupby("y", sort=False):
        idx = grp.index.to_numpy()
        rng = np.random.default_rng(seed + int(y))
        rng.shuffle(idx)
        per_y[int(y)] = np.array_split(idx, k)

    all_idx = set(df.index.tolist())
    folds = []
    for i in range(k):
        val_idx = np.concatenate([per_y[y][i] for y in per_y], axis=0)
        val_set = set(val_idx.tolist())
        train_idx = np.array(sorted(list(all_idx - val_set)), dtype=np.int64)
        val_idx = np.array(sorted(list(val_set)), dtype=np.int64)
        folds.append((train_idx, val_idx))
    return folds

# =========================
# TF.DATA PIPELINE (EXPLICIT)
# =========================
def _normalize(img, mean, std):
    mean = tf.constant(mean, dtype=tf.float32)
    std = tf.constant(std, dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    return (img - mean) / std

def _augment(img, cfg):
    if cfg["AUG_HFLIP"]:
        img = tf.image.random_flip_left_right(img)
    pad = int(cfg["AUG_PAD"])
    if pad > 0:
        img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
        img = tf.image.random_crop(img, size=[cfg["IMG_SIZE"], cfg["IMG_SIZE"], cfg["IN_CHANNELS"]])
    return img

def make_tf_dataset(df: pd.DataFrame, cfg, training: bool):
    paths = df["dst"].to_numpy()
    labels = df["y"].to_numpy().astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, y):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=cfg["IN_CHANNELS"])
        img = tf.image.resize(img, [cfg["IMG_SIZE"], cfg["IMG_SIZE"]], method="bilinear")
        img = tf.cast(img, tf.uint8)  # keep deterministic dtype before aug
        if training:
            img = _augment(img, cfg)
        img = _normalize(img, cfg["NORM_MEAN"], cfg["NORM_STD"])
        return img, y

    if training:
        ds = ds.shuffle(buffer_size=min(len(df), 20000), seed=cfg["SEED"], reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg["BATCH_SIZE"], drop_remainder=training)

    if cfg["PREFETCH"]:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# =========================
# TRAIN / EVAL (EXPLICIT)
# =========================
def train_one_epoch(model, ds, optimizer, loss_fn, cfg):
    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in ds:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
            # add L2 losses (from kernel_regularizer)
            if model.losses:
                loss = loss + tf.add_n(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(y, logits)

    acc = float(train_acc.result().numpy())
    err = 1.0 - acc
    return float(train_loss.result().numpy()), acc, err


def evaluate(model, ds, loss_fn):
    val_loss = tf.keras.metrics.Mean()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in ds:
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        if model.losses:
            loss = loss + tf.add_n(model.losses)

        val_loss.update_state(loss)
        val_acc.update_state(y, logits)

    acc = float(val_acc.result().numpy())
    err = 1.0 - acc
    return float(val_loss.result().numpy()), acc, err

# =========================
# TRAIN / EVAL (EXPLICIT)
# =========================
def train_one_epoch(model, ds, optimizer, loss_fn, cfg):
    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in ds:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
            # add L2 losses (from kernel_regularizer)
            if model.losses:
                loss = loss + tf.add_n(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(y, logits)

    acc = float(train_acc.result().numpy())
    err = 1.0 - acc
    return float(train_loss.result().numpy()), acc, err


def evaluate(model, ds, loss_fn):
    val_loss = tf.keras.metrics.Mean()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in ds:
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        if model.losses:
            loss = loss + tf.add_n(model.losses)

        val_loss.update_state(loss)
        val_acc.update_state(y, logits)

    acc = float(val_acc.result().numpy())
    err = 1.0 - acc
    return float(val_loss.result().numpy()), acc, err


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
