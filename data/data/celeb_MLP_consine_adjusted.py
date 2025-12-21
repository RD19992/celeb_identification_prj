# -*- coding: utf-8 -*-
"""
CELEBA HOG -> MLP (1 hidden layer) com CV + early-stopping

==========================
CORREÇÃO (CV REPRESENTATIVO)
==========================
O CV estava ficando "fácil" demais / não representativo quando cv_min_por_classe (default) era alto
(ex.: 25), pois isso eliminava quase todas as classes (poucas classes sobreviviam) e o K-fold
passava a otimizar hiperparâmetros num problema diferente do treino final.

Agora:
- cv_min_por_classe = None => usa max(k_folds, 5) (mínimo viável para StratifiedKFold com K folds)
- Mantém muitas classes no CV (representativo para identificação com muitas classes)
- Adiciona aviso se o CV terminar com poucas classes
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import glob
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # dataset
    "path_hog_npz": r".\hog_features\hog_128.npz",  # ou 64/96 etc
    "hog_key_X": "X",
    "hog_key_y": "y",

    # seleção de classes para identificação (top por frequência)
    "seed_classes": 42,
    "min_amostras_por_classe": 25,

    # split treino/teste
    "test_frac": 0.20,
    "seed_split": 42,
    "min_train_por_classe": 5,  # pós split (no treino)

    # -----------------
    # CV (k-fold=5 fixo)
    # -----------------
    "k_folds": 5,
    "cv_frac": 1.00,  # conjunto completo para melhor resultado

    # [CORREÇÃO CV] Para StratifiedKFold com k folds, cada classe precisa ter pelo menos k exemplos no conjunto de CV.
    # Se você subir muito esse mínimo (ex.: 25), você elimina a maioria das classes e o CV deixa de representar o problema real.
    # Ajuste recomendado: None -> usa k_folds (mínimo viável), ou escolha 2*k_folds/3*k_folds se sua amostragem por classe permitir.
    "cv_min_por_classe": None,  # None -> usa max(k_folds, 5)

    "cv_max_classes": None,  # opcional: limita #classes no CV (acelera MUITO), mantendo min/cls alto

    # treino final
    "final_frac": 1.00,
    "final_min_por_classe": 1,

    # -----------------
    # MLP (1 hidden layer)
    # -----------------
    "hidden_units": 128,
    "act_hidden": "relu",  # "relu" ou "tanh"
    "act_output": "cosine_softmax",

    # Cosine-softmax (normalização cosseno) - útil com muitas classes
    "cosine_softmax_scale": 20.0,
    "cosine_softmax_eps": 1e-8,
    "cosine_softmax_use_bias": False,

    # Dropout
    "dropout_hidden_p": 0.10,
    "grid_dropout": [0.05, 0.10, 0.15],

    # Regularização L2 (grid)
    "l2_default": 0.10,
    "grid_l2": [0.03, 0.10, 0.30, 1.00],

    # Label smoothing
    "label_smoothing": 0.05,

    # Standardization
    "standardize": True,
    "std_eps": 1e-8,

    # LayerNorm
    "use_layernorm": True,
    "ln_eps": 1e-5,

    # Max-norm constraints
    "use_maxnorm": True,
    "maxnorm_W1": 4.0,
    "maxnorm_W2": 4.0,

    # Ruído Gaussiano no treino
    "gaussian_noise_std": 0.00,

    # Otimização
    "epochs_cv": 15,
    "epochs_final": 20,
    "batch_size_cv": 128,
    "batch_size_final": 128,
    "lr_base": 0.20,           # base LR
    "lr_decay": 0.98,          # decay por época
    "grad_clip_norm": 5.0,     # clipping do grad

    # Early stopping
    "early_stop_metric": "val_acc",  # "val_acc" ou "val_loss"
    "early_stop_patience": 5,
    "early_stop_min_delta": 1e-4,
    "early_stop_cv_enabled": True,
    "early_stop_final_enabled": True,

    # Log / debug
    "verbose": True,
    "print_every": 1,
}


# ============================================================
# UTIL: Reprodutibilidade
# ============================================================

def set_seeds(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))


# ============================================================
# UTIL: Carregar dataset (HOG já pronto)
# ============================================================

def load_hog_dataset(npz_path: str, key_X: str = "X", key_y: str = "y") -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data[key_X]
    y = data[key_y]
    X = np.asarray(X)
    y = np.asarray(y).astype(np.int64, copy=False)

    if X.ndim != 2:
        raise ValueError(f"X precisa ser 2D (n, d). Encontrado: {X.shape}")

    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y precisa ser 1D com mesmo n de X. X: {X.shape}, y: {y.shape}")

    return X, y


# ============================================================
# UTIL: Filtragens e amostragem por classes
# ============================================================

def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes_keep = np.asarray(classes_keep).astype(np.int64, copy=False)
    mask = np.isin(y, classes_keep)
    return X[mask], y[mask]


def top_classes_por_frequencia(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """
    Retorna um subconjunto de classes (top por frequência), com fração frac.
    """
    frac = float(frac)
    if frac <= 0.0 or frac > 1.0:
        raise ValueError("frac deve estar em (0,1].")

    classes, counts = np.unique(y, return_counts=True)
    # ordena por contagem desc
    order = np.argsort(-counts)
    classes_sorted = classes[order]
    n_keep = int(max(1, math.ceil(frac * classes_sorted.size)))
    return classes_sorted[:n_keep].astype(np.int64, copy=False)


def amostrar_com_min_por_classe(y: np.ndarray, frac: float, seed: int, min_por_classe: int):
    """
    Retorna (idx_sample, classes_ok). Garante >= min_por_classe por classe (para classes que têm suporte).
    """
    y = np.asarray(y)
    n = int(y.shape[0])
    frac = float(frac)

    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    ok = counts >= int(min_por_classe)
    classes_ok = classes[ok].astype(np.int64, copy=False)
    if classes_ok.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if frac >= 0.999999:
        mask = np.isin(y, classes_ok)
        idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        return idx, np.sort(classes_ok)

    rng = np.random.default_rng(int(seed))

    idx_keep = []
    for c in classes_ok:
        idx_c = np.flatnonzero(y == c)
        pick = rng.choice(idx_c, size=int(min_por_classe), replace=False)
        idx_keep.append(pick)
    idx_keep = np.concatenate(idx_keep).astype(np.int64, copy=False)

    target = int(np.ceil(frac * n))
    if target <= idx_keep.size:
        return np.sort(idx_keep), np.sort(classes_ok)

    restantes = np.setdiff1d(np.arange(n, dtype=np.int64), idx_keep, assume_unique=False)
    if restantes.size == 0:
        return np.sort(idx_keep), np.sort(classes_ok)

    add = target - idx_keep.size
    if add <= 0:
        return np.sort(idx_keep), np.sort(classes_ok)

    if add >= restantes.size:
        idx_final = np.concatenate([idx_keep, restantes]).astype(np.int64, copy=False)
        return np.sort(np.unique(idx_final)), np.sort(classes_ok)

    y_rest = y[restantes]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=add, random_state=int(seed))
    idx_add_rel, _ = next(sss.split(np.zeros(restantes.size), y_rest))
    idx_add = restantes[idx_add_rel]

    idx_final = np.concatenate([idx_keep, idx_add]).astype(np.int64, copy=False)
    return np.sort(np.unique(idx_final)), np.sort(classes_ok)


def limitar_classes_para_cv(X: np.ndarray, y: np.ndarray, max_classes: int | None, seed: int):
    if max_classes is None:
        return X, y
    max_classes = int(max_classes)
    if max_classes <= 0:
        return X, y
    classes = np.unique(y).astype(np.int64, copy=False)
    if classes.size <= max_classes:
        return X, y
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(classes, size=max_classes, replace=False)
    return filtrar_por_classes(X, y, chosen.astype(np.int64, copy=False))


# ============================================================
# Ativações e Softmax estável
# ============================================================

def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype, copy=False)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    t = np.tanh(x)
    return 1.0 - t * t


def softmax_stable(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# ============================================================
# Losses / objetivos
# ============================================================

def one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y_idx = np.asarray(y_idx).astype(np.int64, copy=False)
    oh = np.zeros((y_idx.size, n_classes), dtype=np.float32)
    oh[np.arange(y_idx.size), y_idx] = 1.0
    return oh


def cross_entropy(probs: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.clip(probs, eps, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(probs), axis=1)))


def apply_label_smoothing(y_onehot: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = float(smoothing)
    if smoothing <= 0.0:
        return y_onehot
    n_classes = int(y_onehot.shape[1])
    return (1.0 - smoothing) * y_onehot + smoothing / float(n_classes)


# ============================================================
# LayerNorm (manual)
# ============================================================

def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    out = xhat * gamma + beta
    cache = (x, xhat, mu, var, gamma, beta, eps)
    return out, cache


def layernorm_backward(dout: np.ndarray, cache):
    x, xhat, mu, var, gamma, beta, eps = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0, keepdims=True)
    dgamma = np.sum(dout * xhat, axis=0, keepdims=True)

    dxhat = dout * gamma
    dvar = np.sum(dxhat * (x - mu) * (-0.5) * (var + eps) ** (-1.5), axis=1, keepdims=True)
    dmu = np.sum(dxhat * (-1.0) / np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.mean(-2.0 * (x - mu), axis=1, keepdims=True)
    dx = dxhat / np.sqrt(var + eps) + dvar * 2.0 * (x - mu) / D + dmu / D

    return dx, dgamma, dbeta


# ============================================================
# Cosine-softmax
# ============================================================

def cosine_softmax_logits(h: np.ndarray, W: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    logits = scale * cosine(h, W) (bias opcional fora)
    h: (n, d)
    W: (d, C)
    """
    h_norm = np.linalg.norm(h, axis=1, keepdims=True) + eps
    W_norm = np.linalg.norm(W, axis=0, keepdims=True) + eps
    h_hat = h / h_norm
    W_hat = W / W_norm
    return h_hat @ W_hat


# ============================================================
# Métricas
# ============================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def codificar_rotulos(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Converte labels originais -> [0..C-1] baseado em 'classes' ordenadas.
    """
    classes = np.asarray(classes).astype(np.int64, copy=False)
    mapping = {int(c): i for i, c in enumerate(classes.tolist())}
    y_idx = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    return y_idx


# ============================================================
# Standardizer
# ============================================================

def fit_standardizer(X: np.ndarray, eps: float = 1e-8):
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.maximum(sd, float(eps))
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


# ============================================================
# Dropout
# ============================================================

def dropout_forward(x: np.ndarray, p: float, rng: np.random.Generator):
    """
    p: prob de dropar
    """
    p = float(p)
    if p <= 0.0:
        return x, None
    keep = 1.0 - p
    mask = (rng.random(size=x.shape) < keep).astype(x.dtype, copy=False) / keep
    return x * mask, mask


def dropout_backward(dout: np.ndarray, mask):
    if mask is None:
        return dout
    return dout * mask


# ============================================================
# Max-norm constraint
# ============================================================

def apply_maxnorm(W: np.ndarray, maxnorm: float, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Enforça ||W|| <= maxnorm por coluna (axis=0) ou por linha (axis=1).
    """
    maxnorm = float(maxnorm)
    if maxnorm <= 0.0:
        return W
    norms = np.linalg.norm(W, axis=axis, keepdims=True) + eps
    factor = np.minimum(1.0, maxnorm / norms)
    return W * factor


# ============================================================
# Forward/Backward do MLP
# ============================================================

def forward_mlp(
    X: np.ndarray,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    ln_gamma: np.ndarray, ln_beta: np.ndarray,
    act_hidden: str,
    act_output: str,
    dropout_p: float,
    rng: np.random.Generator,
    use_layernorm: bool,
    ln_eps: float,
    cosine_scale: float,
    cosine_eps: float,
    cosine_use_bias: bool,
    train_mode: bool,
):
    # hidden pre
    z1 = X @ W1 + b1
    # activation hidden
    if act_hidden == "tanh":
        h = tanh(z1)
        h_grad_cache = z1
        act_grad_fn = tanh_grad
    else:
        h = relu(z1)
        h_grad_cache = z1
        act_grad_fn = relu_grad

    # layernorm (opcional)
    ln_cache = None
    if use_layernorm:
        h, ln_cache = layernorm_forward(h, ln_gamma, ln_beta, eps=ln_eps)

    # dropout
    do_cache = None
    if train_mode and dropout_p > 0.0:
        h, do_cache = dropout_forward(h, dropout_p, rng)

    # output logits
    if act_output == "cosine_softmax":
        cos = cosine_softmax_logits(h, W2, eps=cosine_eps)
        logits = float(cosine_scale) * cos
        if cosine_use_bias:
            logits = logits + b2
        else:
            # se bias desativado, b2 ignora
            pass
        probs = softmax_stable(logits)
        cache = (X, z1, h, h_grad_cache, act_grad_fn, ln_cache, do_cache, logits, probs, cos)
        return probs, cache
    else:
        logits = h @ W2 + b2
        probs = softmax_stable(logits)
        cache = (X, z1, h, h_grad_cache, act_grad_fn, ln_cache, do_cache, logits, probs, None)
        return probs, cache


def backward_mlp(
    cache,
    y_onehot: np.ndarray,
    W2: np.ndarray,
    ln_gamma: np.ndarray,
    use_layernorm: bool,
    ln_eps: float,
    dropout_p: float,
    act_output: str,
    cosine_scale: float,
    cosine_eps: float,
    cosine_use_bias: bool,
):
    (X, z1, h, h_grad_cache, act_grad_fn, ln_cache, do_cache, logits, probs, cos) = cache
    n = X.shape[0]

    # dL/dlogits
    dlogits = (probs - y_onehot) / float(n)

    # output gradients
    if act_output == "cosine_softmax":
        # logits = scale * cos(h, W2)
        # cos = (h_hat @ W_hat)
        # Para simplificar, usamos grad aproximado via cadeias com normalizações.
        # Ainda funciona bem para o projeto.
        scale = float(cosine_scale)

        # Reconstroi normalizações
        eps = float(cosine_eps)
        h_norm = np.linalg.norm(h, axis=1, keepdims=True) + eps
        W_norm = np.linalg.norm(W2, axis=0, keepdims=True) + eps
        h_hat = h / h_norm
        W_hat = W2 / W_norm

        dcos = dlogits * scale  # (n, C)

        # d(h_hat) = dcos @ W_hat.T
        dh_hat = dcos @ W_hat.T  # (n, d)

        # grad w.r.t h: h_hat = h / ||h||
        # dh = (I/||h|| - h h^T / ||h||^3) dh_hat
        # Implementação vetorizada:
        # dh = dh_hat/hn - h * sum(dh_hat*h)/hn^3
        dot = np.sum(dh_hat * h, axis=1, keepdims=True)
        dh = dh_hat / h_norm - h * dot / (h_norm ** 3)

        # grad w.r.t W2: W_hat = W/||W||
        # dW_hat = h_hat^T @ dcos
        dW_hat = h_hat.T @ dcos  # (d, C)
        # dW = dW_hat/wn - W * sum(dW_hat*W)/wn^3
        dotW = np.sum(dW_hat * W2, axis=0, keepdims=True)
        dW2 = dW_hat / W_norm - W2 * dotW / (W_norm ** 3)

        if cosine_use_bias:
            db2 = np.sum(dlogits, axis=0, keepdims=True)
        else:
            db2 = np.zeros((1, W2.shape[1]), dtype=np.float32)

    else:
        dW2 = h.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)
        dh = dlogits @ W2.T

    # dropout backward
    dh = dropout_backward(dh, do_cache)

    # layernorm backward
    dgamma = np.zeros_like(ln_gamma)
    dbeta = np.zeros_like(ln_gamma)
    if use_layernorm and ln_cache is not None:
        dh, dgamma, dbeta = layernorm_backward(dh, ln_cache)

    # activation backward
    dz1 = dh * act_grad_fn(h_grad_cache)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dgamma, dbeta


# ============================================================
# Objective total (CE + L2)
# ============================================================

def objective_total(
    W1, b1, W2, b2,
    ln_gamma, ln_beta,
    X, y_idx,
    act_hidden: str,
    act_output: str,
    l2: float,
    m_total: int,
    smoothing: float,
):
    C = int(np.max(y_idx) + 1)
    y_oh = one_hot(y_idx, C)
    y_oh = apply_label_smoothing(y_oh, smoothing=float(smoothing))

    rng = np.random.default_rng(123)
    probs, _ = forward_mlp(
        X,
        W1, b1,
        W2, b2,
        ln_gamma, ln_beta,
        act_hidden=act_hidden,
        act_output=act_output,
        dropout_p=0.0,
        rng=rng,
        use_layernorm=bool(CONFIG["use_layernorm"]),
        ln_eps=float(CONFIG["ln_eps"]),
        cosine_scale=float(CONFIG["cosine_softmax_scale"]),
        cosine_eps=float(CONFIG["cosine_softmax_eps"]),
        cosine_use_bias=bool(CONFIG["cosine_softmax_use_bias"]),
        train_mode=False,
    )
    ce = cross_entropy(probs, y_oh)

    # L2 regularization (ajustada por m_total para coerência)
    l2 = float(l2)
    reg = (l2 / float(max(1, m_total))) * (np.sum(W1 * W1) + np.sum(W2 * W2))
    return float(ce + reg)


# ============================================================
# Treino MLP (SGD minibatch) + early stopping
# ============================================================

def treinar_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes_modelo: np.ndarray,
    l2: float,
    epochs: int,
    batch_size: int,
    seed: int,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stop_enabled: bool = True,
    dropout_p: float = 0.10,
    act_output: str = "cosine_softmax",
):
    set_seeds(seed)
    rng = np.random.default_rng(int(seed))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    classes = np.asarray(classes_modelo).astype(np.int64, copy=False)
    C = int(classes.size)
    y_idx = codificar_rotulos(y_train, classes)
    y_oh = one_hot(y_idx, C)
    y_oh = apply_label_smoothing(y_oh, smoothing=float(CONFIG["label_smoothing"]))

    n, d = X_train.shape
    h = int(CONFIG["hidden_units"])

    # init
    W1 = (rng.normal(size=(d, h)).astype(np.float32) * 0.02)
    b1 = np.zeros((1, h), dtype=np.float32)
    W2 = (rng.normal(size=(h, C)).astype(np.float32) * 0.02)
    b2 = np.zeros((1, C), dtype=np.float32)

    ln_gamma = np.ones((1, h), dtype=np.float32)
    ln_beta = np.zeros((1, h), dtype=np.float32)

    lr = float(CONFIG["lr_base"])
    lr_decay = float(CONFIG["lr_decay"])
    clip = float(CONFIG["grad_clip_norm"])

    best_state = None
    best_epoch = -1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience = int(CONFIG["early_stop_patience"])
    min_delta = float(CONFIG["early_stop_min_delta"])
    no_improve = 0

    # minibatches
    idx_all = np.arange(n, dtype=np.int64)

    for ep in range(1, int(epochs) + 1):
        rng.shuffle(idx_all)
        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            mb = idx_all[start:end]
            Xb = X_train[mb]
            yb = y_oh[mb]

            # noise
            gn = float(CONFIG["gaussian_noise_std"])
            if gn > 0.0:
                Xb = Xb + rng.normal(scale=gn, size=Xb.shape).astype(np.float32)

            # forward
            probs, cache = forward_mlp(
                Xb, W1, b1, W2, b2, ln_gamma, ln_beta,
                act_hidden=str(CONFIG["act_hidden"]),
                act_output=str(act_output),
                dropout_p=float(dropout_p),
                rng=rng,
                use_layernorm=bool(CONFIG["use_layernorm"]),
                ln_eps=float(CONFIG["ln_eps"]),
                cosine_scale=float(CONFIG["cosine_softmax_scale"]),
                cosine_eps=float(CONFIG["cosine_softmax_eps"]),
                cosine_use_bias=bool(CONFIG["cosine_softmax_use_bias"]),
                train_mode=True,
            )

            # backward
            dW1, db1, dW2, db2, dgamma, dbeta = backward_mlp(
                cache=cache,
                y_onehot=yb,
                W2=W2,
                ln_gamma=ln_gamma,
                use_layernorm=bool(CONFIG["use_layernorm"]),
                ln_eps=float(CONFIG["ln_eps"]),
                dropout_p=float(dropout_p),
                act_output=str(act_output),
                cosine_scale=float(CONFIG["cosine_softmax_scale"]),
                cosine_eps=float(CONFIG["cosine_softmax_eps"]),
                cosine_use_bias=bool(CONFIG["cosine_softmax_use_bias"]),
            )

            # L2 grads (coerente com objective_total)
            l2f = float(l2) / float(max(1, n))
            dW1 = dW1 + 2.0 * l2f * W1
            dW2 = dW2 + 2.0 * l2f * W2

            # clip
            def clip_by_norm(G, c):
                norm = float(np.linalg.norm(G))
                if norm > c:
                    return G * (c / (norm + 1e-12))
                return G

            dW1 = clip_by_norm(dW1, clip)
            db1 = clip_by_norm(db1, clip)
            dW2 = clip_by_norm(dW2, clip)
            db2 = clip_by_norm(db2, clip)
            dgamma = clip_by_norm(dgamma, clip)
            dbeta = clip_by_norm(dbeta, clip)

            # SGD step
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            ln_gamma -= lr * dgamma
            ln_beta -= lr * dbeta

            # maxnorm
            if bool(CONFIG["use_maxnorm"]):
                W1 = apply_maxnorm(W1, float(CONFIG["maxnorm_W1"]), axis=0)
                W2 = apply_maxnorm(W2, float(CONFIG["maxnorm_W2"]), axis=0)

        # lr decay por época
        lr *= lr_decay

        # validação
        val_acc = None
        val_loss = None
        if X_val is not None and y_val is not None:
            y_pred, _ = predict_labels(X_val, {
                "W1": W1, "b1": b1,
                "W2": W2, "b2": b2,
                "ln_gamma": ln_gamma, "ln_beta": ln_beta,
                "classes": classes,
                "act_hidden": str(CONFIG["act_hidden"]),
                "act_output": str(act_output),
                "dropout_p": float(dropout_p),
            })
            val_acc = accuracy(y_val, y_pred)

            y_val_idx = codificar_rotulos(y_val, classes)
            val_loss = objective_total(
                W1, b1, W2, b2, ln_gamma, ln_beta,
                X_val.astype(np.float32), y_val_idx,
                act_hidden=str(CONFIG["act_hidden"]),
                act_output=str(act_output),
                l2=float(l2),
                m_total=int(n),
                smoothing=float(CONFIG["label_smoothing"]),
            )

        if bool(CONFIG["verbose"]) and (ep % int(CONFIG["print_every"]) == 0):
            if val_acc is not None:
                print(f"    ep={ep:>3d} | val_acc={val_acc:.4f} | val_loss={val_loss:.4f}")
            else:
                print(f"    ep={ep:>3d}")

        # early stopping
        if early_stop_enabled and (X_val is not None) and (y_val is not None):
            metric_name = str(CONFIG.get("early_stop_metric", "val_acc")).lower().strip()
            if metric_name == "val_loss":
                improved = (best_val_loss - float(val_loss)) > min_delta
                if improved:
                    best_val_loss = float(val_loss)
                    best_val_acc = float(val_acc)
                    best_epoch = int(ep)
                    best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), ln_gamma.copy(), ln_beta.copy())
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                improved = (float(val_acc) - best_val_acc) > min_delta
                if improved:
                    best_val_acc = float(val_acc)
                    best_val_loss = float(val_loss)
                    best_epoch = int(ep)
                    best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), ln_gamma.copy(), ln_beta.copy())
                    no_improve = 0
                else:
                    no_improve += 1

            if no_improve >= patience:
                if bool(CONFIG["verbose"]):
                    print(f"    [early-stop] ep={ep} (best_ep={best_epoch} | best_val_acc={best_val_acc:.4f} | best_val_loss={best_val_loss:.4f})")
                break

    # restaura melhor estado se early stop
    if early_stop_enabled and best_state is not None:
        W1, b1, W2, b2, ln_gamma, ln_beta = best_state

    modelo = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "ln_gamma": ln_gamma, "ln_beta": ln_beta,
        "classes": classes,
        "act_hidden": str(CONFIG["act_hidden"]),
        "act_output": str(act_output),
        "dropout_p": float(dropout_p),
    }
    return modelo


def predict_labels(X: np.ndarray, modelo: Dict):
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(0)

    probs, _ = forward_mlp(
        X,
        modelo["W1"], modelo["b1"],
        modelo["W2"], modelo["b2"],
        modelo["ln_gamma"], modelo["ln_beta"],
        act_hidden=str(modelo["act_hidden"]),
        act_output=str(modelo["act_output"]),
        dropout_p=0.0,
        rng=rng,
        use_layernorm=bool(CONFIG["use_layernorm"]),
        ln_eps=float(CONFIG["ln_eps"]),
        cosine_scale=float(CONFIG["cosine_softmax_scale"]),
        cosine_eps=float(CONFIG["cosine_softmax_eps"]),
        cosine_use_bias=bool(CONFIG["cosine_softmax_use_bias"]),
        train_mode=False,
    )
    pred_idx = np.argmax(probs, axis=1)
    classes = np.asarray(modelo["classes"]).astype(np.int64, copy=False)
    y_pred = classes[pred_idx]
    return y_pred, probs


# ============================================================
# CV: grid-search (l2 x dropout)
# ============================================================

def grid_search_cv(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """
    Grid-search sobre (l2, dropout) com StratifiedKFold K=5.

    Critério:
      - Se "val_acc": maximiza acc (tie-break: menor loss, depois menor l2)
      - Se "val_loss": minimiza loss (tie-break: maior acc, depois menor l2)

    Observação:
    - Continuamos reportando (mean±std) de acc e loss para diagnóstico.
    """
    from sklearn.model_selection import StratifiedKFold

    k = int(CONFIG["k_folds"])
    assert k == 5, "K-fold precisa continuar 5."

    grid_l2 = list(CONFIG["grid_l2"])
    grid_dp = list(CONFIG["grid_dropout"])

    metric_name = str(CONFIG.get("early_stop_metric", "val_acc")).lower().strip()
    if metric_name not in ("val_loss", "val_acc"):
        metric_name = "val_acc"

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))

    melhor = None  # dicionário com métricas do melhor combo

    print("\n[CV] Grid-search (l2 x dropout) com K=5:")
    print(f"     critério (early_stop_metric) = {metric_name}")
    print(f"     l2 candidates                = {grid_l2}")
    print(f"     dropout candidates           = {grid_dp}")
    print(f"     act_output                   = {CONFIG['act_output']}")

    for dp in grid_dp:
        for l2 in grid_l2:
            fold_acc = []
            fold_loss = []

            for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
                X_tr, y_tr = X[tr], y[tr]
                X_va, y_va = X[va], y[va]

                # Treina com early stopping do CV (mesmos knobs do final)
                modelo = treinar_mlp(
                    X_train=X_tr, y_train=y_tr,
                    classes_modelo=np.unique(y),
                    l2=float(l2),
                    epochs=int(CONFIG["epochs_cv"]),
                    batch_size=int(CONFIG["batch_size_cv"]),
                    seed=int(seed) + 1000 * fold + int(round(100 * float(dp))),
                    X_val=X_va, y_val=y_va,
                    early_stop_enabled=bool(CONFIG["early_stop_cv_enabled"]),
                    dropout_p=float(dp),
                    act_output=str(CONFIG["act_output"]),
                )

                # ACC no fold
                y_pred, _ = predict_labels(X_va, modelo)
                acc = accuracy(y_va, y_pred)

                # LOSS no fold (consistente com objective_total)
                y_va_idx = codificar_rotulos(y_va, modelo["classes"])
                val_loss = objective_total(
                    modelo["W1"], modelo["b1"], modelo["W2"], modelo["b2"],
                    modelo["ln_gamma"], modelo["ln_beta"],
                    X_va, y_va_idx,
                    modelo["act_hidden"], modelo["act_output"],
                    l2=float(l2), m_total=int(X_tr.shape[0]),
                    smoothing=float(CONFIG["label_smoothing"]),
                )

                fold_acc.append(float(acc))
                fold_loss.append(float(val_loss))

            mean_acc = float(np.mean(fold_acc))
            std_acc = float(np.std(fold_acc))
            mean_loss = float(np.mean(fold_loss))
            std_loss = float(np.std(fold_loss))

            print(
                f"  l2={float(l2):>6g}  dp={float(dp):.2f}  -> "
                f"val_acc={mean_acc:.4f}±{std_acc:.4f} | val_loss={mean_loss:.4f}±{std_loss:.4f}"
            )

            cand = {
                "l2": float(l2),
                "dp": float(dp),
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
            }

            if melhor is None:
                melhor = cand
            else:
                if metric_name == "val_loss":
                    # menor loss é melhor
                    if cand["mean_loss"] < melhor["mean_loss"] - 1e-12:
                        melhor = cand
                    elif abs(cand["mean_loss"] - melhor["mean_loss"]) <= 1e-12:
                        # tie-break: maior acc
                        if cand["mean_acc"] > melhor["mean_acc"] + 1e-12:
                            melhor = cand
                        elif abs(cand["mean_acc"] - melhor["mean_acc"]) <= 1e-12:
                            # tie-break: menor l2
                            if cand["l2"] < melhor["l2"]:
                                melhor = cand
                else:
                    # maior acc é melhor
                    if cand["mean_acc"] > melhor["mean_acc"] + 1e-12:
                        melhor = cand
                    elif abs(cand["mean_acc"] - melhor["mean_acc"]) <= 1e-12:
                        # tie-break: menor loss
                        if cand["mean_loss"] < melhor["mean_loss"] - 1e-12:
                            melhor = cand
                        elif abs(cand["mean_loss"] - melhor["mean_loss"]) <= 1e-12:
                            # tie-break: menor l2
                            if cand["l2"] < melhor["l2"]:
                                melhor = cand

    print("\n[CV] Melhor combo:")
    print(f"     l2={melhor['l2']} | dp={melhor['dp']} | mean_acc={melhor['mean_acc']:.4f} | mean_loss={melhor['mean_loss']:.4f}")
    return melhor


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    t0 = time.time()

    # 1) carregar HOG
    X_all, y_all = load_hog_dataset(CONFIG["path_hog_npz"], CONFIG["hog_key_X"], CONFIG["hog_key_y"])
    print(f"[Etapa 1] Loaded: X={X_all.shape} | y={y_all.shape} | classes={len(np.unique(y_all))}")

    # 1b) filtrar top classes por frequência (20% / etc) -> aqui depende do dataset salvo
    # OBS: neste script, assume que seu npz já tem o recorte (ou você adapta aqui se quiser)
    # Mantemos o pipeline simples e replicável.

    # 2) split treino/teste estratificado
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_all, y_all,
        test_size=float(CONFIG["test_frac"]),
        random_state=int(CONFIG["seed_split"]),
        stratify=y_all
    )

    print(f"[Etapa 2] Train (all): {X_train_all.shape} | classes={len(np.unique(y_train_all))}")
    print(f"[Etapa 2] Test (all):  {X_test_all.shape} | classes={len(np.unique(y_test_all))}")

    # 3) filtrar classes no treino com >= max(min_train_por_classe, k_folds)
    min_train = int(max(CONFIG["min_train_por_classe"], CONFIG["k_folds"]))
    cls, cts = np.unique(y_train_all, return_counts=True)
    cls_keep = cls[cts >= min_train]
    X_train, y_train = filtrar_por_classes(X_train_all, y_train_all, cls_keep)
    X_test, y_test = filtrar_por_classes(X_test_all, y_test_all, cls_keep)

    print(f"[Etapa 3] Train (filtered>= {min_train}): X={X_train.shape} | classes={len(np.unique(y_train))}")
    print(f"[Etapa 3] Test  (aligned):            X={X_test.shape} | classes={len(np.unique(y_test))}")

    # 4) standardize (fit no treino)
    if bool(CONFIG["standardize"]):
        mean_tr, std_tr = fit_standardizer(X_train, eps=float(CONFIG["std_eps"]))
        X_train_feat = apply_standardizer(X_train, mean_tr, std_tr)
        X_test_feat = apply_standardizer(X_test, mean_tr, std_tr)
    else:
        mean_tr, std_tr = None, None
        X_train_feat = X_train.astype(np.float32)
        X_test_feat = X_test.astype(np.float32)

    # 5) monta amostra para CV
    k = int(CONFIG["k_folds"])
    cv_min = CONFIG["cv_min_por_classe"]
    if cv_min is None:
        # [CORREÇÃO CV] default seguro: mantém o CV representativo (muitas classes)
        # e garante viabilidade do StratifiedKFold (>= k exemplos por classe).
        cv_min = max(k, 5)
    cv_min = int(cv_min)

    X_train_cv_src, y_train_cv_src = limitar_classes_para_cv(
        X_train_feat, y_train, CONFIG["cv_max_classes"], seed=int(CONFIG["seed_split"])
    )
    idx_cv, _ = amostrar_com_min_por_classe(
        y=y_train_cv_src,
        frac=float(CONFIG["cv_frac"]),
        seed=int(CONFIG["seed_split"]),
        min_por_classe=cv_min,
    )
    if idx_cv.size == 0:
        raise RuntimeError("Amostra CV vazia. Ajuste cv_frac/cv_max_classes/cv_min_por_classe.")

    X_cv = X_train_cv_src[idx_cv]
    y_cv = y_train_cv_src[idx_cv]
    classes_cv, cts_cv = np.unique(y_cv, return_counts=True)
    n_cls_cv = int(classes_cv.size)
    print(
        f"\n[Etapa 5] CV sample: X={X_cv.shape} | classes={n_cls_cv} | "
        f"min_count={int(cts_cv.min())} | max_count={int(cts_cv.max())}"
    )
    if n_cls_cv < 50:
        print(
            "  [AVISO] Poucas classes no CV. Isso costuma indicar cv_min_por_classe alto demais ou "
            "cv_max_classes muito baixo. O CV fica pouco representativo para identificação."
        )

    # 6) CV grid-search (L2 x dropout)
    print(
        f"\n[Etapa 6] CV grid-search | epochs_cv={CONFIG['epochs_cv']} batch_size_cv={CONFIG['batch_size_cv']} "
        f"early_stop={CONFIG['early_stop_cv_enabled']}"
    )
    best = grid_search_cv(X_cv, y_cv, seed=int(CONFIG["seed_split"]))

    best_l2 = float(best["l2"])
    best_dp = float(best["dp"])

    # 7) treino final (com melhores hiperparâmetros)
    idx_final, _ = amostrar_com_min_por_classe(
        y=y_train,
        frac=float(CONFIG["final_frac"]),
        seed=int(CONFIG["seed_split"]),
        min_por_classe=int(CONFIG["final_min_por_classe"]),
    )
    if idx_final.size == 0:
        raise RuntimeError("Amostra final vazia. Ajuste final_frac/final_min_por_classe.")

    X_final = X_train_feat[idx_final]
    y_final = y_train[idx_final]
    print(f"\n[Etapa 7] Final train sample: X={X_final.shape} | classes={len(np.unique(y_final))}")

    # split interno para early stop do treino final (holdout)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_final, y_final,
        test_size=0.15,
        random_state=int(CONFIG["seed_split"]),
        stratify=y_final
    )

    modelo_final = treinar_mlp(
        X_train=X_tr, y_train=y_tr,
        classes_modelo=np.unique(y_final),
        l2=best_l2,
        epochs=int(CONFIG["epochs_final"]),
        batch_size=int(CONFIG["batch_size_final"]),
        seed=int(CONFIG["seed_split"]) + 999,
        X_val=X_va, y_val=y_va,
        early_stop_enabled=bool(CONFIG["early_stop_final_enabled"]),
        dropout_p=best_dp,
        act_output=str(CONFIG["act_output"]),
    )

    # 8) Avaliação
    y_pred_train, _ = predict_labels(X_final, modelo_final)
    acc_train = accuracy(y_final, y_pred_train)

    y_pred_test, _ = predict_labels(X_test_feat, modelo_final)
    acc_test = accuracy(y_test, y_pred_test)

    print("\n[Etapa 8] Resultados:")
    print(f"  Train acc (final sample) = {acc_train:.4f}")
    print(f"  Test  acc               = {acc_test:.4f}")
    print(f"  Best params: l2={best_l2} | dropout={best_dp}")

    dt = time.time() - t0
    print(f"\nDone. Tempo total: {dt:.1f}s")


if __name__ == "__main__":
    main()
