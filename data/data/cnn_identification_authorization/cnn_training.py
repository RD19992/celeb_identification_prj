# -*- coding: utf-8 -*-
"""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Treinamento de CNN (ResNet "explicit") com K-Fold CV (sem sklearn).

Arquivos gerados (por execução), dentro de:
  <DATASET_DIR>/runs/<YYYYMMDD_HHMMSS>/

- run_errors.csv            (métricas por fold/época)
- summary.json              (agregados e melhores erros por fold)
- hyperparameters.json      (CONFIG usado na execução)
- label2idx.json            (mapeamento label_original -> y_reindexado)
- [opcional] models/*.keras (salvo apenas se SAVE_MODEL=True)

Obs.: SAVE_MODEL começa DESABILITADO por padrão.
"""

from __future__ import annotations

from pathlib import Path
import csv
import datetime as _dt
import json
import math
import random
import time
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

# =========================
# GPU TEST
# =========================

# Use só a primeira GPU DirectML (normalmente a discreta)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("[INFO] Using GPU:", gpus[0])
    except Exception as e:
        print("[WARN] Could not set GPU config:", e)
else:
    print("[INFO] No GPU found, using CPU.")

# =========================
# CONFIGURAÇÕES
# =========================
CONFIG: Dict[str, Any] = {
    # Data produced by ingestion script
    "DATASET_DIR": Path(__file__).resolve().parent / "celeba_rgb_256x256",
    "MANIFEST_NAME": "manifest.csv",
    "ONLY_OK": True,                 # use ok==True if column exists

    # Class filtering
    "TOP_CLASS_FRACTION": 0.01,       # top 1% most frequent classes
    "KFOLDS": 3,
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

    # Outputs
    "RUNS_DIRNAME": "runs",
    "SAVE_MODEL": False,              # <-- começa desabilitado
}


# =========================
# UTIL
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def now_stamp() -> str:
    # YYYYMMDD_HHMMSS
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    def _default(x):
        if isinstance(x, Path):
            return str(x)
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


# =========================
# RESNET (EXPLICIT)
# =========================
def conv_bn_act(
    x,
    filters: int,
    k: int,
    stride: int,
    use_bn: bool,
    activation: str | None,
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


def build_min_resnet(cfg: Dict[str, Any], num_classes: int) -> Model:
    """
    Minimal ResNet (explicit):
      stem -> stages -> GAP -> Dense(logits)
    """
    inp = layers.Input(
        shape=(cfg["IMG_SIZE"], cfg["IMG_SIZE"], cfg["IN_CHANNELS"]),
        name="input"
    )

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
def load_and_filter_manifest(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Lê manifest, filtra top classes por frequência, e remapeia labels para y em [0..C-1].
    Também remove classes que não tenham >= KFOLDS exemplos (necessário para CV estratificado).
    """
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

    if df.empty:
        raise RuntimeError("No valid images after filtering by file existence / ok flag.")

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
    kfolds = int(cfg["KFOLDS"])
    counts = df["y"].value_counts()
    keep_y = set(counts[counts >= kfolds].index.tolist())
    df = df[df["y"].isin(keep_y)].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise RuntimeError("Dataset empty after class filtering for k-fold.")

    print(f"[INFO] images={len(df)} classes={df['y'].nunique()}")
    return df, label2idx


def stratified_kfold_indices(df: pd.DataFrame, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def _augment(img, cfg: Dict[str, Any]):
    if cfg["AUG_HFLIP"]:
        img = tf.image.random_flip_left_right(img)
    pad = int(cfg["AUG_PAD"])
    if pad > 0:
        img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
        img = tf.image.random_crop(img, size=[cfg["IMG_SIZE"], cfg["IMG_SIZE"], cfg["IN_CHANNELS"]])
    return img


def make_tf_dataset(df: pd.DataFrame, cfg: Dict[str, Any], training: bool):
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
def train_one_epoch(model: Model, ds, optimizer, loss_fn):
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


def evaluate(model: Model, ds, loss_fn):
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
# K-FOLD CV RUNNER
# =========================
def run_kfold_cv(cfg: Dict[str, Any]) -> Tuple[float, Path]:
    """
    Executa K-Fold CV e grava arquivos de log.
    Retorna: (cv_mean_best_val_err, run_dir)
    """
    set_seed(int(cfg["SEED"]))

    df, label2idx = load_and_filter_manifest(cfg)
    k = int(cfg["KFOLDS"])
    folds = stratified_kfold_indices(df, k=k, seed=int(cfg["SEED"]))
    num_classes = int(df["y"].nunique())

    # output dir
    stamp = now_stamp()
    run_dir = cfg["DATASET_DIR"] / cfg["RUNS_DIRNAME"] / stamp
    ensure_dir(run_dir)

    # save hyperparams + label map
    save_json(cfg, run_dir / "hyperparameters.json")
    save_json({str(k): int(v) for k, v in label2idx.items()}, run_dir / "label2idx.json")

    # CSV metrics file
    errors_csv = run_dir / "run_errors.csv"
    with errors_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fold", "epoch",
            "train_loss", "train_acc", "train_err",
            "val_loss", "val_acc", "val_err",
            "epoch_seconds"
        ])

        fold_summaries = []
        best_val_errs = []

        for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
            tf.keras.backend.clear_session()
            # variar seed por fold, mantendo reprodutível
            set_seed(int(cfg["SEED"]) + 1000 * fold_i)

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_ds = make_tf_dataset(train_df, cfg, training=True)
            val_ds = make_tf_dataset(val_df, cfg, training=False)

            with tf.device(cfg["DEVICE"]):
                model = build_min_resnet(cfg, num_classes=num_classes)

            optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg["LR"]))
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            best_val_err = float("inf")
            best_epoch = -1
            best_weights = None

            print(f"\n[INFO] Fold {fold_i}/{k} | train={len(train_df)} val={len(val_df)} | classes={num_classes}")

            for epoch in range(1, int(cfg["EPOCHS"]) + 1):
                t0 = time.time()

                tr_loss, tr_acc, tr_err = train_one_epoch(model, train_ds, optimizer, loss_fn)
                va_loss, va_acc, va_err = evaluate(model, val_ds, loss_fn)

                dt = time.time() - t0
                w.writerow([
                    fold_i, epoch,
                    f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_err:.6f}",
                    f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_err:.6f}",
                    f"{dt:.3f}"
                ])
                f.flush()

                print(
                    f"[Fold {fold_i}][Epoch {epoch}/{cfg['EPOCHS']}] "
                    f"train_loss={tr_loss:.4f} train_err={tr_err:.4f} | "
                    f"val_loss={va_loss:.4f} val_err={va_err:.4f} | {dt:.1f}s"
                )

                if va_err < best_val_err:
                    best_val_err = float(va_err)
                    best_epoch = int(epoch)
                    best_weights = model.get_weights()

            # restore best weights for this fold (best val_err)
            if best_weights is not None:
                model.set_weights(best_weights)

            best_val_errs.append(best_val_err)
            fold_summaries.append({
                "fold": fold_i,
                "best_epoch": best_epoch,
                "best_val_err": best_val_err,
            })

            # optional model saving
            if bool(cfg.get("SAVE_MODEL", False)):
                models_dir = run_dir / "models"
                ensure_dir(models_dir)
                save_path = models_dir / f"model_fold{fold_i}_best.keras"
                model.save(save_path)
                print(f"[SAVED] {save_path}")

    # summary
    mean_err = float(np.mean(best_val_errs))
    std_err = float(np.std(best_val_errs, ddof=1)) if len(best_val_errs) > 1 else 0.0

    summary = {
        "kfolds": k,
        "num_images": int(len(df)),
        "num_classes": int(num_classes),
        "best_val_errs": best_val_errs,
        "cv_mean_best_val_err": mean_err,
        "cv_std_best_val_err": std_err,
        "folds": fold_summaries,
        "run_dir": str(run_dir),
        "device": cfg["DEVICE"],
        "timestamp": stamp,
    }
    save_json(summary, run_dir / "summary.json")

    print("\n[DONE] CV finished.")
    print(f"[RESULT] mean(best_val_err)={mean_err:.6f} | std={std_err:.6f}")
    print(f"[FILES] {run_dir}")
    return mean_err, run_dir


# =========================
# MAIN
# =========================
def main() -> float:
    """
    Treina ResNet 'explicit' com K-Fold CV.
    Retorna o erro médio (val_err) dos melhores epochs por fold.
    """
    print("[INFO] Using device:", CONFIG["DEVICE"])
    mean_err, _run_dir = run_kfold_cv(CONFIG)
    return mean_err


if __name__ == "__main__":
    main()
