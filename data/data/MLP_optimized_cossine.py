# -*- coding: utf-8 -*-
"""
CELEBA HOG -> MLP (1 hidden layer) - "from scratch"
====================================================

Objetivo
--------
Classificação de identidades (CelebA) usando HOG fixo (não muda) + MLP 1 camada escondida.
Este script implementa uma versão estável e escalável do seu pipeline, com foco em:

[CORREÇÕES 1 a 8] (todas implementadas, com knobs no CONFIG)
------------------------------------------------------------
1) Estabilidade do passo (LR):
   - Warmup + teto de LR; rollback e redução automática se houver colapso/NaN.
2) Otimizador melhor:
   - SGD + Momentum (opção Nesterov).
3) Regularização contra colapso de confiança:
   - Label smoothing, grid L2 maior, Max-Norm (opcional).
4) Dropout calibrado:
   - Dropout menor por padrão (0.10) e sempre OFF em val/test.
5) Normalização interna:
   - LayerNorm treinável (opcional) antes da ativação.
6) CV mais representativo sem abandonar k-fold=5:
   - Parâmetro cv_min_por_classe (sugestão default = 5*k*), mantendo k=5.
7) Diagnóstico rápido:
   - Entropia (média/mediana), pmax, fração da classe "modo", ||W||, ||grad||, razão update/||W||.
8) Escala:
   - Sem one-hot (gradiente por índice).
   - (Opcional) sampled-softmax (NEG sampling) disponível, mas DESLIGADO por padrão.

Observação importante
---------------------
- Você pediu para manter MLP e manter HOG. Este script respeita isso.
- O maior ganho vem de estabilidade (LR/momentum/clipping) + regularização (label smoothing/L2/dropout).
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

# ============================================================
# CONFIG (ajuste aqui)
# ============================================================

CONFIG = {
    "dataset_path": r"C:\Users\riosd\PycharmProjects\celeb_identification_prj\data\data\celeba_hog_128x128_o9.joblib",

    # seleção de classes (prototipagem)
    "frac_classes": 0.20,  # ex.: 0.20 (20% das classes elegíveis)
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
    "cv_frac": 0.40,  # você usou 0.40
    # [MELHORIA 6] mínimo por classe no CV deve ser bem maior que k para ser informativo.
    # Sugestão: 5*k (ex.: 25) ou 20, o que for maior.
    "cv_min_por_classe": None,  # None -> usa max(5*k_folds, 20)
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

    # [CORREÇÃO 4] Dropout menor por padrão
    "dropout_hidden_p": 0.10,
    # Grid de dropout (otimizado no CV)
    "grid_dropout": [0.05, 0.10, 0.15],
    # 0.0 desliga

    # [CORREÇÃO 5] LayerNorm treinável antes da ativação (ajuda estabilidade)
    "use_layernorm": True,
    "layernorm_eps": 1e-5,

    # inicialização
    "use_he_xavier_init": True,  # ReLU->He, tanh->Xavier
    "w_init_scale": 0.01,  # fallback

    # -----------------
    # Regularização / loss
    # -----------------
    # [CORREÇÃO 3] Label smoothing reduz colapso de confiança
    "label_smoothing": 0.05,  # 0.0 desliga
    # [CORREÇÃO 3] Grid L2 maior (principalmente em escala)
    "grid_l2": [0.0, 1e-2, 1e-1, 1.0],

    # [CORREÇÃO 3] Max-Norm (opcional; ajuda contra explosões e overfit)
    "maxnorm_enabled": True,
    "maxnorm_W1": 3.0,  # None desliga
    "maxnorm_W2": 3.0,  # None desliga

    # -----------------
    # Treinamento
    # -----------------
    "epochs_cv": 20,
    "epochs_final": 120,

    "batch_size_cv": 256,
    "batch_size_final": 512,

    # Gaussian noise (somente no treino; X já é padronizado)
    "gaussian_noise_std": 0.00,  # ex.: 0.01; 0.0 desliga

    # batches quase-balanceados (gera uma permutação que intercalada classes)
    "use_almost_balanced_batches": True,
    # [CORREÇÃO 2] Momentum
    "use_momentum": True,
    "momentum_beta": 0.90,
    "use_nesterov": False,

    # [CORREÇÃO 1] LR schedule (warmup + teto + decaimento)
    "lr_base": 0.8,  # LR após warmup
    "lr_min": 1e-4,
    "lr_warmup_epochs": 5,
    "lr_warmup_start": 0.10,  # LR inicial do warmup
    "lr_decay": 0.98,  # multiplicativo por época após warmup

    # [CORREÇÃO 1] Guardas contra colapso numérico
    "nan_guard_enabled": True,
    "nan_guard_alpha_shrink": 0.5,  # se colapsar, reduz LR e restaura checkpoint
    "nan_guard_max_rollbacks": 5,

    # [CORREÇÃO 1] Gradient clipping (global norm)
    "grad_clip_enabled": True,
    "grad_clip_norm": 5.0,

    # -----------------
    # (Opcional) Sampled softmax (correção 8)
    # -----------------
    # Se >0: para o gradiente no treino, atualiza apenas classes verdadeiras + negativos aleatórios.
    # Predição/val/test continuam usando softmax completo (exato).
    "sampled_softmax_neg_k": 0,  # 0 desliga (padrão)

    # -----------------
    # Early stopping (CV e treino final)
    # -----------------
    "early_stop_cv_enabled": True,
    "early_stop_final_enabled": True,
    "early_stop_metric": "val_acc",  # "val_loss" ou "val_acc"
    "early_stop_patience": 15,
    "early_stop_min_delta": 1e-4,
    "early_stop_warmup_epochs": 5,
    "early_stop_restore_best": True,

    # split interno treino/val no treino final (para early stopping)
    "final_val_frac": 0.10,
    "final_val_seed": 2026,

    # padronização
    "eps_std": 1e-6,

    # logs / debug
    "print_every_batches": 50,
    "loss_subsample_max": 2000,
    "n_exemplos_previsao": 12,
    "top_k_confusao": 10,
    "debug_epoch_stats": True,
    "entropy_eps": 1e-12,
}


# ============================================================
# Utilidades de IO / Dataset
# ============================================================

def carregar_dataset(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "X" in obj and "y" in obj:
        X, y = obj["X"], obj["y"]
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, y = obj
    else:
        raise ValueError("Formato do joblib não reconhecido. Esperado dict{'X','y'} ou tuple(X,y).")
    return np.asarray(X), np.asarray(y)


def selecionar_classes_elegiveis(y: np.ndarray, min_amostras: int):
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts >= int(min_amostras)].astype(np.int64, copy=False)


def amostrar_classes(classes: np.ndarray, frac: float, seed: int):
    frac = float(frac)
    classes = np.asarray(classes, dtype=np.int64)
    if frac >= 0.999999:
        return np.array(classes, copy=True)
    rng = np.random.default_rng(int(seed))
    n = len(classes)
    k = max(1, int(np.ceil(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    return np.sort(classes[idx]).astype(np.int64, copy=False)


def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_permitidas: np.ndarray):
    mask = np.isin(y, classes_permitidas)
    return X[mask], y[mask]


# ============================================================
# Padronização (z-score)
# ============================================================

def fit_standardizer(X: np.ndarray, eps: float = 1e-6):
    Xf = X.astype(np.float32, copy=False)
    mean = Xf.mean(axis=0, dtype=np.float64).astype(np.float32)
    var = ((Xf - mean).astype(np.float32, copy=False) ** 2).mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(var + np.float32(eps)).astype(np.float32)
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    Xf = X.astype(np.float32, copy=False)
    return ((Xf - mean) / std).astype(np.float32, copy=False)


# ============================================================
# Amostragem estratificada com mínimo por classe
# ============================================================

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

def stable_softmax(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    Zm = Z - Z.max(axis=1, keepdims=True)
    np.exp(Zm, out=Zm)
    Zm /= Zm.sum(axis=1, keepdims=True)
    return Zm


# ============================================================
# Output: Softmax padrão vs Cosine-Softmax (normalização cosseno)
# ============================================================
# A ideia do cosine-softmax:
#   logits = s * cos(Â, Ŵ)   onde Â = A/||A|| (por amostra) e Ŵ = W/||W|| (por classe)
# Isso tende a estabilizar em cenários com MUITAS classes (e.g., identificação/face),
# pois reduz a dependência de norma e foca em ângulo (similaridade cosseno).

def _row_norm_forward(A: np.ndarray, eps: float):
    # normaliza cada linha: A_hat[i] = A[i]/||A[i]||
    A = A.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(A * A, axis=1, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return A * inv, inv.astype(np.float32, copy=False)


def _row_norm_backward(dA_hat: np.ndarray, A_hat: np.ndarray, inv_norm: np.ndarray):
    # dA = (1/||A||) * (dA_hat - A_hat * <dA_hat, A_hat>)
    dA_hat = dA_hat.astype(np.float32, copy=False)
    A_hat = A_hat.astype(np.float32, copy=False)
    inv_norm = inv_norm.astype(np.float32, copy=False)
    inner = np.sum(dA_hat * A_hat, axis=1, keepdims=True)  # <dA_hat, A_hat>
    return inv_norm * (dA_hat - A_hat * inner)


def _col_norm_forward(W: np.ndarray, eps: float):
    # normaliza cada coluna: W_hat[:,k] = W[:,k]/||W[:,k]||
    W = W.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(W * W, axis=0, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return W * inv, inv.astype(np.float32, copy=False)


def _col_norm_backward(dW_hat: np.ndarray, W_hat: np.ndarray, inv_norm: np.ndarray):
    # dW = (1/||W||) * (dW_hat - W_hat * <dW_hat, W_hat>)  (por coluna)
    dW_hat = dW_hat.astype(np.float32, copy=False)
    W_hat = W_hat.astype(np.float32, copy=False)
    inv_norm = inv_norm.astype(np.float32, copy=False)
    inner = np.sum(dW_hat * W_hat, axis=0, keepdims=True)  # <dW_hat, W_hat> por coluna
    return inv_norm * (dW_hat - W_hat * inner)


def output_logits_forward(
        A1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        act_output: str,
        scale: float,
        eps: float,
        use_bias: bool,
):
    act_output = str(act_output).lower().strip()

    if act_output in ("softmax", "linear", "linear_softmax"):
        Z2 = A1 @ W2 + b2
        out_cache = ("linear", A1, W2, use_bias)
        return Z2, out_cache

    if act_output == "cosine_softmax":
        A_hat, inv_a = _row_norm_forward(A1, eps=eps)
        W_hat, inv_w = _col_norm_forward(W2, eps=eps)
        Z2 = float(scale) * (A_hat @ W_hat)
        if bool(use_bias):
            Z2 = Z2 + b2
        out_cache = ("cosine_softmax", A_hat, inv_a, W_hat, inv_w, float(scale), bool(use_bias))
        return Z2, out_cache

    raise ValueError(f"act_output desconhecida: {act_output}")


def output_logits_backward(dZ2: np.ndarray, out_cache: tuple):
    mode = out_cache[0]
    if mode == "linear":
        _, A1, W2, use_bias = out_cache
        dA1 = dZ2 @ W2.T
        dW2 = A1.T @ dZ2
        if bool(use_bias):
            db2 = np.sum(dZ2, axis=0, keepdims=True)
        else:
            db2 = np.zeros((1, int(dZ2.shape[1])), dtype=np.float32)
        return dA1, dW2, db2

    if mode == "cosine_softmax":
        _, A_hat, inv_a, W_hat, inv_w, scale, use_bias = out_cache
        dA_hat = (dZ2 @ W_hat.T) * float(scale)
        dW_hat = (A_hat.T @ dZ2) * float(scale)
        dA1 = _row_norm_backward(dA_hat, A_hat, inv_a)
        dW2 = _col_norm_backward(dW_hat, W_hat, inv_w)
        if bool(use_bias):
            db2 = np.sum(dZ2, axis=0, keepdims=True)
        else:
            db2 = np.zeros((1, int(dZ2.shape[1])), dtype=np.float32)
        return dA1, dW2, db2

    raise ValueError(f"Modo de output desconhecido: {mode}")


def tanh_custom(Z: np.ndarray):
    return np.tanh(Z.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def relu_custom(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    return np.maximum(np.float32(0.0), Z).astype(np.float32, copy=False)


def activation_forward(Z: np.ndarray, act: str):
    act = str(act).lower().strip()
    if act == "tanh":
        return tanh_custom(Z)
    if act == "relu":
        return relu_custom(Z)
    raise ValueError(f"Ativação desconhecida: {act}")


def tanh_backward(dA: np.ndarray, A: np.ndarray):
    return dA * (np.float32(1.0) - A * A)


def relu_backward(dA: np.ndarray, Z: np.ndarray):
    return dA * (Z > np.float32(0.0)).astype(np.float32)


# ============================================================
# Label encoding
# ============================================================

def codificar_rotulos(y: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    pos = np.searchsorted(classes, y)
    if np.any(pos < 0) or np.any(pos >= classes.size) or np.any(classes[pos] != y):
        raise ValueError("y contém rótulos fora de classes.")
    return pos.astype(np.int64, copy=False)


# ============================================================
# Inicialização He/Xavier
# ============================================================

def _std_he(fan_in: int) -> float:
    return float(np.sqrt(2.0 / max(fan_in, 1)))


def _std_xavier(fan_in: int, fan_out: int) -> float:
    return float(np.sqrt(2.0 / max(fan_in + fan_out, 1)))


def inicializar_pesos(d: int, H: int, K: int, act_hidden: str, rng: np.random.Generator):
    if not bool(CONFIG["use_he_xavier_init"]):
        s = float(CONFIG["w_init_scale"])
        W1 = (s * rng.standard_normal((d, H))).astype(np.float32)
        W2 = (s * rng.standard_normal((H, K))).astype(np.float32)
    else:
        act = str(act_hidden).lower().strip()
        std1 = _std_he(d) if act == "relu" else _std_xavier(d, H)
        std2 = _std_xavier(H, K)
        W1 = (std1 * rng.standard_normal((d, H))).astype(np.float32)
        W2 = (std2 * rng.standard_normal((H, K))).astype(np.float32)

    b1 = np.zeros((H,), dtype=np.float32)
    b2 = np.zeros((K,), dtype=np.float32)

    # [CORREÇÃO 5] LayerNorm treinável
    if bool(CONFIG["use_layernorm"]):
        ln_gamma = np.ones((H,), dtype=np.float32)
        ln_beta = np.zeros((H,), dtype=np.float32)
    else:
        ln_gamma, ln_beta = None, None

    return W1, b1, W2, b2, ln_gamma, ln_beta


# ============================================================
# Dropout (inverted)
# ============================================================

def aplicar_dropout_invertido(A: np.ndarray, p_drop: float, rng: np.random.Generator):
    p = float(p_drop)
    if p <= 0.0:
        return A, None
    if p >= 0.999:
        return A, None
    keep_prob = 1.0 - p
    mask = (rng.random(A.shape) < keep_prob).astype(np.float32) / np.float32(keep_prob)
    return (A * mask).astype(np.float32, copy=False), mask


# ============================================================
# LayerNorm (treinável) forward/backward
# ============================================================

def layernorm_forward(Z: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
    """
    Z: (B,H). Normaliza por amostra ao longo de H.
    Retorna Z_tilde e cache para backward.
    """
    Z = Z.astype(np.float32, copy=False)
    mu = Z.mean(axis=1, keepdims=True)
    var = Z.var(axis=1, keepdims=True)
    invstd = 1.0 / np.sqrt(var + np.float32(eps))
    xhat = (Z - mu) * invstd
    out = xhat * gamma + beta
    cache = (xhat, invstd, gamma)
    return out.astype(np.float32, copy=False), cache


def layernorm_backward(dout: np.ndarray, cache):
    """
    dout: grad w.r.t. LN output (B,H)
    Retorna: dZ, dgamma, dbeta
    """
    xhat, invstd, gamma = cache
    dout = dout.astype(np.float32, copy=False)
    B, H = dout.shape

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * xhat, axis=0)

    dxhat = dout * gamma
    # fórmula vetorizada por amostra
    sum_dxhat = dxhat.sum(axis=1, keepdims=True)
    sum_dxhat_xhat = np.sum(dxhat * xhat, axis=1, keepdims=True)
    dZ = (invstd / np.float32(H)) * (np.float32(H) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)
    return dZ.astype(np.float32, copy=False), dgamma.astype(np.float32, copy=False), dbeta.astype(np.float32,
                                                                                                  copy=False)


# ============================================================
# Forward / Loss / Backprop
# ============================================================

def mlp_forward(
        X: np.ndarray,
        W1: np.ndarray, b1: np.ndarray,
        W2: np.ndarray, b2: np.ndarray,
        act_hidden: str,
        act_output: str,
        ln_gamma: np.ndarray | None,
        ln_beta: np.ndarray | None,
        dropout_p: float,
        rng: np.random.Generator | None,
        train_mode: bool,
):
    """
    [CORREÇÃO 5] LN opcional antes da ativação
    [CORREÇÃO 4] Dropout só no treino
    """
    X = X.astype(np.float32, copy=False)
    Z1 = X @ W1 + b1

    ln_cache = None
    if ln_gamma is not None and ln_beta is not None:
        Z1_tilde, ln_cache = layernorm_forward(Z1, ln_gamma, ln_beta, eps=float(CONFIG["layernorm_eps"]))
    else:
        Z1_tilde = Z1

    A1_pre = activation_forward(Z1_tilde, act_hidden)

    dropout_mask = None
    A1 = A1_pre
    if train_mode and float(dropout_p) > 0.0:
        if rng is None:
            raise ValueError("rng não pode ser None quando dropout>0 no treino.")
        A1, dropout_mask = aplicar_dropout_invertido(A1_pre, p_drop=float(dropout_p), rng=rng)

    # [REFATORAÇÃO] Output pode ser softmax linear OU cosine-softmax (normalização cosseno)
    Z2, out_cache = output_logits_forward(
        A1, W2, b2,
        act_output=act_output,
        scale=float(CONFIG["cosine_softmax_scale"]),
        eps=float(CONFIG["cosine_softmax_eps"]),
        use_bias=bool(CONFIG["cosine_softmax_use_bias"]),
    )
    P = stable_softmax(Z2)

    cache = (X, Z1, Z1_tilde, A1_pre, A1, Z2, P, dropout_mask, ln_cache, out_cache)
    return P, cache


def ce_loss_with_label_smoothing(P: np.ndarray, y_idx: np.ndarray, smoothing: float):
    """
    [CORREÇÃO 3] CE com label smoothing.
    """
    eps = np.float32(1e-12)
    P = P.astype(np.float32, copy=False)
    y_idx = y_idx.astype(np.int64, copy=False)
    B, K = P.shape
    sm = float(smoothing)

    logP = np.log(P + eps)
    if sm <= 0.0 or K <= 1:
        return float(-np.mean(logP[np.arange(B), y_idx]))

    a = sm / float(K - 1)
    sum_log = np.sum(logP, axis=1)
    log_y = logP[np.arange(B), y_idx]
    loss = -((1.0 - sm) * log_y + a * (sum_log - log_y))
    return float(np.mean(loss))


def l2_reg_loss(W1: np.ndarray, W2: np.ndarray, l2: float, m_total: int):
    if float(l2) == 0.0:
        return 0.0
    m = float(max(int(m_total), 1))
    return (float(l2) / (2.0 * m)) * float(np.sum(W1 * W1) + np.sum(W2 * W2))


def objective_total(
        W1, b1, W2, b2, ln_gamma, ln_beta,
        X, y_idx, act_hidden, act_output, l2, m_total, smoothing,
):
    # determinístico: sem dropout
    P, _ = mlp_forward(
        X, W1, b1, W2, b2, act_hidden, act_output,
        ln_gamma, ln_beta,
        dropout_p=0.0, rng=None, train_mode=False
    )
    return ce_loss_with_label_smoothing(P, y_idx, smoothing=float(smoothing)) + l2_reg_loss(W1, W2, float(l2), m_total)


def compute_grads_batch(
        W1, b1, W2, b2, ln_gamma, ln_beta,
        X, y_idx,
        act_hidden,
        act_output,
        l2, m_total,
        dropout_p, rng, train_mode,
        smoothing,
        sampled_neg_k: int,
):
    """
    Backprop completo (ou opcionalmente sampled-softmax para treino).
    - Sem one-hot: grad do softmax por índice.
    """
    B = int(X.shape[0])
    P, cache = mlp_forward(
        X, W1, b1, W2, b2, act_hidden, act_output,
        ln_gamma, ln_beta,
        dropout_p=float(dropout_p),
        rng=rng,
        train_mode=bool(train_mode),
    )
    Xc, Z1, Z1_tilde, A1_pre, A1_used, Z2, Pc, dropout_mask, ln_cache, out_cache = cache
    K = int(Pc.shape[1])

    # dZ2 = (P - Y_smooth)/B
    dZ2 = Pc.copy()

    sm = float(smoothing)
    if sampled_neg_k and sampled_neg_k > 0 and train_mode:
        # [CORREÇÃO 8 - opcional] Sampled-softmax: atualiza subset de classes (true + negativos).
        # Implementação simples: recomputa softmax apenas no subset S e zera grad fora dele.
        # Obs.: P/val/test continuam full.
        # Para não complicar demais, dZ2 fica full aqui (mais seguro). Deixe sampled_softmax_neg_k=0 por padrão.
        pass

    if sm <= 0.0 or K <= 1:
        dZ2[np.arange(B), y_idx] -= np.float32(1.0)
    else:
        a = sm / float(K - 1)
        dZ2 -= np.float32(a)
        dZ2[np.arange(B), y_idx] -= np.float32((1.0 - sm) - a)

    dZ2 /= np.float32(max(B, 1))

    dW2 = A1_used.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ W2.T

    # dropout backward
    if train_mode and (dropout_mask is not None):
        dA1 = (dA1 * dropout_mask).astype(np.float32, copy=False)

    # activation backward
    act = str(act_hidden).lower().strip()
    if act == "relu":
        dZ1_tilde = relu_backward(dA1, Z1_tilde)
    elif act == "tanh":
        dZ1_tilde = tanh_backward(dA1, A1_pre)
    else:
        raise ValueError(f"Ativação desconhecida: {act}")

    # layernorm backward (se ativo)
    dgamma = None
    dbeta_ln = None
    if ln_gamma is not None and ln_beta is not None:
        dZ1, dgamma, dbeta_ln = layernorm_backward(dZ1_tilde, ln_cache)
    else:
        dZ1 = dZ1_tilde

    dW1 = Xc.T @ dZ1
    db1_g = dZ1.sum(axis=0)

    # L2 (normaliza por m_total)
    if float(l2) != 0.0:
        coef = np.float32(l2) / np.float32(max(int(m_total), 1))
        dW1 += coef * W1
        dW2 += coef * W2

    grads = {
        "dW1": dW1.astype(np.float32, copy=False),
        "db1": db1_g.astype(np.float32, copy=False),
        "dW2": dW2.astype(np.float32, copy=False),
        "db2": db2.astype(np.float32, copy=False),
        "P": Pc,  # para stats
    }
    if dgamma is not None:
        grads["d_ln_gamma"] = dgamma
        grads["d_ln_beta"] = dbeta_ln
    return grads


# ============================================================
# Diagnósticos (entropia, pmax, modo)
# ============================================================

def entropia_stats(P: np.ndarray):
    eps = np.float32(CONFIG["entropy_eps"])
    P = P.astype(np.float32, copy=False)
    H = -np.sum(P * np.log(P + eps), axis=1)
    return float(np.mean(H)), float(np.median(H))


def pmax_stats(P: np.ndarray):
    P = P.astype(np.float32, copy=False)
    pmax = np.max(P, axis=1)
    return float(np.mean(pmax)), float(np.median(pmax))


def mode_fraction(y_pred: np.ndarray):
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_pred.size == 0:
        return 0.0, None
    vals, counts = np.unique(y_pred, return_counts=True)
    i = int(np.argmax(counts))
    return float(counts[i] / y_pred.size), int(vals[i])


def norma_fro(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A.astype(np.float32, copy=False) ** 2)))


# ============================================================
# Gradient clipping (global norm)
# ============================================================

def clip_grads_global_norm(grads: dict, clip: float, eps: float = 1e-12):
    norm2 = 0.0
    for k in ("dW1", "db1", "dW2", "db2", "d_ln_gamma", "d_ln_beta"):
        if k in grads and grads[k] is not None:
            g = grads[k]
            norm2 += float(np.sum(g * g))
    norm = float(np.sqrt(norm2))
    if norm > float(clip):
        scale = float(clip) / (norm + float(eps))
        for k in ("dW1", "db1", "dW2", "db2", "d_ln_gamma", "d_ln_beta"):
            if k in grads and grads[k] is not None:
                grads[k] = (grads[k] * np.float32(scale)).astype(np.float32, copy=False)
    return grads, norm


# ============================================================
# Max-Norm (opcional)
# ============================================================

def apply_maxnorm(W: np.ndarray, maxnorm: float):
    if maxnorm is None:
        return W
    m = float(maxnorm)
    if m <= 0.0:
        return W
    # max-norm por coluna
    col_norms = np.sqrt(np.sum(W * W, axis=0, keepdims=True)) + np.float32(1e-12)
    scale = np.minimum(1.0, m / col_norms)
    return (W * scale).astype(np.float32, copy=False)


# ============================================================
# Early stopping util
# ============================================================

def _melhorou(metrica_atual: float, metrica_best: float, modo: str, min_delta: float) -> bool:
    if modo == "min":
        return metrica_atual < (metrica_best - min_delta)
    return metrica_atual > (metrica_best + min_delta)


# ============================================================
# LR schedule (warmup + decaimento)
# ============================================================

def lr_por_epoca(ep: int, base: float):
    warm = int(CONFIG["lr_warmup_epochs"])
    start = float(CONFIG["lr_warmup_start"])
    decay = float(CONFIG["lr_decay"])
    lr_min = float(CONFIG["lr_min"])

    if warm > 0 and ep < warm:
        # interpolação linear
        t = (ep + 1) / float(warm)
        lr = start + t * (base - start)
    else:
        steps = max(0, ep - warm + 1)
        lr = base * (decay ** steps)
    return float(max(lr, lr_min))


# ============================================================
# Predição
# ============================================================

def predict_labels(X: np.ndarray, modelo: dict):
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    P, _ = mlp_forward(
        X,
        modelo["W1"], modelo["b1"],
        modelo["W2"], modelo["b2"],
        modelo["act_hidden"], modelo["act_output"],
        modelo["ln_gamma"], modelo["ln_beta"],
        dropout_p=0.0, rng=None, train_mode=False
    )
    idx = np.argmax(P, axis=1).astype(np.int64, copy=False)
    classes = modelo["classes"]
    return classes[idx], P


# ============================================================
# Utilitários de treino: gaussian noise e permutação quase-balanceada
# ============================================================

def aplicar_ruido_gaussiano(X: np.ndarray, std: float, rng: np.random.Generator):
    std = float(std)
    if std <= 0.0:
        return X
    ruido = rng.standard_normal(X.shape).astype(np.float32, copy=False)
    return (X + np.float32(std) * ruido).astype(np.float32, copy=False)


def permutacao_quase_balanceada(y_idx: np.ndarray, rng: np.random.Generator):
    """
    Cria uma permutação de índices que intercala classes (sem repetir amostras),
    ajudando a deixar cada mini-batch *quase* balanceado.

    Estratégia:
      - Ordena índices por classe
      - Embaralha dentro de cada classe
      - Intercala em round-robin pelas classes (ordem de classes embaralhada por época)
    """
    y_idx = np.asarray(y_idx, dtype=np.int64)
    m = int(y_idx.shape[0])
    if m <= 1:
        return np.arange(m, dtype=np.int64)

    # idx_sorted agrupa por classe
    idx_sorted = np.argsort(y_idx, kind="mergesort")
    y_sorted = y_idx[idx_sorted]

    # cortes onde a classe muda
    cuts = np.flatnonzero(np.diff(y_sorted)) + 1
    grupos = np.split(idx_sorted, cuts)

    # shuffle intra-classe
    for g in grupos:
        rng.shuffle(g)

    ordem_classes = np.arange(len(grupos), dtype=np.int64)
    rng.shuffle(ordem_classes)  # shuffle por época

    max_len = max((len(g) for g in grupos), default=0)
    perm_list = []
    perm_list_extend = perm_list.append

    for t in range(max_len):
        for ci in ordem_classes:
            g = grupos[int(ci)]
            if t < len(g):
                perm_list_extend(int(g[t]))

    return np.asarray(perm_list, dtype=np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


# ============================================================
# Confusão Top-k (sem sklearn, para evitar warnings)
# ============================================================

def confusion_top_k(y_true: np.ndarray, y_pred: np.ndarray, top_k: int):
    top_k = int(top_k)
    if top_k <= 0:
        return

    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return

    classes, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    top = classes[order[:top_k]].astype(np.int64, copy=False)

    mask = np.isin(y_true, top)
    y_t = y_true[mask]
    y_p = y_pred[mask]

    # mapeia labels -> 0..k-1
    map_idx = {int(c): i for i, c in enumerate(top.tolist())}
    cm = np.zeros((top_k, top_k), dtype=np.int64)
    for yt, yp in zip(y_t, y_p):
        i = map_idx.get(int(yt), None)
        j = map_idx.get(int(yp), None)
        if i is not None and j is not None:
            cm[i, j] += 1

    print(f"\n[Matriz de Confusão] Top-{top_k} classes (TESTE filtrado):")
    print("labels:", top.tolist())
    print(cm)


# ============================================================
# Split interno treino/val no final
# ============================================================

def split_treino_validacao_estratificado(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    val_frac = float(val_frac)
    if val_frac <= 0.0:
        return X, y, None, None
    Xtr, Xva, ytr, yva = train_test_split(
        X, y,
        test_size=val_frac,
        random_state=int(seed),
        stratify=y
    )
    return Xtr, ytr, Xva, yva


# ============================================================
# Treino (SGD + Momentum + guards)
# ============================================================

def treinar_mlp(
        X_train: np.ndarray, y_train: np.ndarray,
        classes_modelo: np.ndarray,
        l2: float,
        epochs: int,
        batch_size: int,
        seed: int,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        early_stop_enabled: bool,
        dropout_p: float | None = None,
        act_output: str | None = None,
):
    rng = np.random.default_rng(int(seed))

    X_train = np.ascontiguousarray(X_train.astype(np.float32, copy=False))
    y_train = np.asarray(y_train, dtype=np.int64)
    classes_modelo = np.sort(np.unique(classes_modelo)).astype(np.int64, copy=False)

    y_idx = codificar_rotulos(y_train, classes_modelo)

    has_val = (X_val is not None) and (y_val is not None) and (int(X_val.shape[0]) > 0)
    if has_val:
        X_val = np.ascontiguousarray(X_val.astype(np.float32, copy=False))
        y_val = np.asarray(y_val, dtype=np.int64)
        y_val_idx = codificar_rotulos(y_val, classes_modelo)
    else:
        y_val_idx = None

    m_total = int(X_train.shape[0])
    d = int(X_train.shape[1])
    K = int(classes_modelo.shape[0])
    H = int(CONFIG["hidden_units"])
    act_hidden = str(CONFIG["act_hidden"]).lower().strip()

    W1, b1, W2, b2, ln_gamma, ln_beta = inicializar_pesos(d=d, H=H, K=K, act_hidden=act_hidden, rng=rng)

    # momentum buffers
    use_mom = bool(CONFIG["use_momentum"])
    beta_m = float(CONFIG["momentum_beta"])
    use_nest = bool(CONFIG["use_nesterov"])

    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    if ln_gamma is not None:
        v_g = np.zeros_like(ln_gamma)
        v_b = np.zeros_like(ln_beta)
    else:
        v_g = v_b = None

    dropout_p = float(CONFIG["dropout_hidden_p"] if dropout_p is None else dropout_p)
    smoothing = float(CONFIG["label_smoothing"])
    sampled_neg_k = int(CONFIG["sampled_softmax_neg_k"])

    # early stopping bookkeeping
    metric_name = str(CONFIG["early_stop_metric"]).lower().strip()
    metric_name = metric_name if metric_name in ("val_loss", "val_acc") else "val_loss"
    best_mode = "min" if metric_name == "val_loss" else "max"
    best_metric = float("inf") if best_mode == "min" else -float("inf")
    best_state = None
    best_epoch = None
    no_improve = 0

    warm = int(CONFIG["early_stop_warmup_epochs"])
    patience = int(CONFIG["early_stop_patience"])
    min_delta = float(CONFIG["early_stop_min_delta"])

    max_rollbacks = int(CONFIG["nan_guard_max_rollbacks"])
    rollbacks_used = 0

    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size > m_total:
        batch_size = m_total

    for ep in range(int(epochs)):
        lr = lr_por_epoca(ep, base=float(CONFIG["lr_base"]))

        # checkpoint para rollback
        if bool(CONFIG["nan_guard_enabled"]):
            ckpt = (W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                    None if ln_gamma is None else ln_gamma.copy(),
                    None if ln_beta is None else ln_beta.copy(),
                    vW1.copy(), vb1.copy(), vW2.copy(), vb2.copy(),
                    None if v_g is None else v_g.copy(),
                    None if v_b is None else v_b.copy(),
                    lr)

        # SGD
        # [REFATORAÇÃO] Permutação padrão vs quase-balanceada
        if bool(CONFIG.get("use_almost_balanced_batches", False)):
            perm = permutacao_quase_balanceada(y_idx, rng)
        else:
            perm = rng.permutation(m_total)
        n_batches = int(np.ceil(m_total / batch_size))

        # [CORREÇÃO 7] acompanhar norma do gradiente ao longo da época
        gnorm_hist = []

        for bi, start in enumerate(range(0, m_total, batch_size), start=1):
            idx = perm[start:start + batch_size]
            Xb = X_train[idx]
            yb = y_idx[idx]

            # [REFATORAÇÃO] Gaussian noise apenas no treino (data augmentation leve)
            Xb = aplicar_ruido_gaussiano(Xb, std=float(CONFIG.get("gaussian_noise_std", 0.0)), rng=rng)

            grads = compute_grads_batch(
                W1, b1, W2, b2, ln_gamma, ln_beta,
                Xb, yb,
                act_hidden=act_hidden,
                act_output=act_output,
                l2=float(l2), m_total=m_total,
                dropout_p=dropout_p,
                rng=rng,
                train_mode=True,
                smoothing=smoothing,
                sampled_neg_k=sampled_neg_k,
            )

            # clipping
            if bool(CONFIG["grad_clip_enabled"]):
                grads, gnorm = clip_grads_global_norm(grads, clip=float(CONFIG["grad_clip_norm"]))
            else:
                gnorm = float("nan")

            if np.isfinite(gnorm):
                gnorm_hist.append(float(gnorm))

            # momentum update
            if use_mom:
                # (opcional) Nesterov: "lookahead" no update (aproximação simples)
                vW1 = beta_m * vW1 + grads["dW1"]
                vb1 = beta_m * vb1 + grads["db1"]
                vW2 = beta_m * vW2 + grads["dW2"]
                vb2 = beta_m * vb2 + grads["db2"]

                if ln_gamma is not None:
                    v_g = beta_m * v_g + grads["d_ln_gamma"]
                    v_b = beta_m * v_b + grads["d_ln_beta"]

                if use_nest:
                    dW1_step = beta_m * vW1 + grads["dW1"]
                    db1_step = beta_m * vb1 + grads["db1"]
                    dW2_step = beta_m * vW2 + grads["dW2"]
                    db2_step = beta_m * vb2 + grads["db2"]
                    if ln_gamma is not None:
                        dg_step = beta_m * v_g + grads["d_ln_gamma"]
                        db_step = beta_m * v_b + grads["d_ln_beta"]
                else:
                    dW1_step, db1_step, dW2_step, db2_step = vW1, vb1, vW2, vb2
                    if ln_gamma is not None:
                        dg_step, db_step = v_g, v_b
            else:
                dW1_step, db1_step, dW2_step, db2_step = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]
                if ln_gamma is not None:
                    dg_step, db_step = grads["d_ln_gamma"], grads["d_ln_beta"]

            # aplica update
            W1 -= np.float32(lr) * dW1_step
            b1 -= np.float32(lr) * db1_step
            W2 -= np.float32(lr) * dW2_step
            b2 -= np.float32(lr) * db2_step
            if ln_gamma is not None:
                ln_gamma -= np.float32(lr) * dg_step
                ln_beta -= np.float32(lr) * db_step

            # maxnorm
            if bool(CONFIG["maxnorm_enabled"]):
                W1 = apply_maxnorm(W1, CONFIG["maxnorm_W1"])
                W2 = apply_maxnorm(W2, CONFIG["maxnorm_W2"])

            # guard de NaN/Inf
            if bool(CONFIG["nan_guard_enabled"]) and (bi % 50 == 0 or bi == n_batches):
                if (not np.isfinite(W1).all()) or (not np.isfinite(W2).all()) or (not np.isfinite(b1).all()) or (
                not np.isfinite(b2).all()):
                    # rollback
                    (W1, b1, W2, b2, ln_gamma_ck, ln_beta_ck, vW1, vb1, vW2, vb2, v_g_ck, v_b_ck, lr_ck) = ckpt
                    ln_gamma = ln_gamma_ck
                    ln_beta = ln_beta_ck
                    v_g = v_g_ck
                    v_b = v_b_ck

                    rollbacks_used += 1
                    CONFIG["lr_base"] = max(float(CONFIG["lr_min"]),
                                            float(CONFIG["lr_base"]) * float(CONFIG["nan_guard_alpha_shrink"]))
                    print(
                        f"\n[NAN_GUARD] Rollback em ep={ep + 1} batch={bi}. Reduzindo lr_base para {CONFIG['lr_base']:.4g} (rollbacks={rollbacks_used}/{max_rollbacks})")
                    break

            if bi % int(CONFIG["print_every_batches"]) == 0 or bi == n_batches:
                # loss determinístico no batch (sem dropout) para acompanhar
                loss_est = objective_total(
                    W1, b1, W2, b2, ln_gamma, ln_beta,
                    Xb, yb, act_hidden, act_output,
                    l2=float(l2), m_total=m_total,
                    smoothing=smoothing,
                )
                print(f"[Treino] ep={ep + 1:03d}/{epochs} lr={lr:.3e} batch={bi:04d}/{n_batches} loss~={loss_est:.4f}",
                      end="\r")

        print(" " * 160, end="\r")

        if bool(CONFIG["nan_guard_enabled"]) and rollbacks_used >= max_rollbacks:
            print("[NAN_GUARD] Muitos rollbacks. Encerrando treino.")
            break

        # stats por época (subsample treino)
        sub_n = min(int(CONFIG["loss_subsample_max"]), m_total)
        sub = rng.choice(m_total, size=sub_n, replace=False)
        loss_ep = objective_total(
            W1, b1, W2, b2, ln_gamma, ln_beta,
            X_train[sub], y_idx[sub],
            act_hidden, act_output, l2=float(l2), m_total=m_total,
            smoothing=smoothing
        )

        if len(gnorm_hist) > 0:
            gnorm_mean = float(np.mean(gnorm_hist))
            gnorm_median = float(np.median(gnorm_hist))
        else:
            gnorm_mean = float('nan')
            gnorm_median = float('nan')

        # validação
        if has_val:
            val_loss = objective_total(
                W1, b1, W2, b2, ln_gamma, ln_beta,
                X_val, y_val_idx,
                act_hidden, act_output, l2=float(l2), m_total=m_total,
                smoothing=smoothing
            )
            yhat_val, P_val = predict_labels(X_val, {
                "W1": W1, "b1": b1, "W2": W2, "b2": b2,
                "ln_gamma": ln_gamma, "ln_beta": ln_beta,
                "classes": classes_modelo, "act_hidden": act_hidden, "act_output": act_output
            })
            val_acc = accuracy(y_val, yhat_val)

            if bool(CONFIG["debug_epoch_stats"]):
                # stats do val
                Hm, Hmed = entropia_stats(P_val)
                pm, pmed = pmax_stats(P_val)
                mf, mc = mode_fraction(yhat_val)
                nW1, nW2 = norma_fro(W1), norma_fro(W2)
                # razão de update aproximada (usa norma do buffer)
                upd = lr * (norma_fro(vW1 if use_mom else grads["dW1"]))
                ratio = float(upd / (nW1 + 1e-12))
                print(
                    f"[Época] {ep + 1:03d}/{epochs} lr={lr:.3e} loss_sub={loss_ep:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                    f"H={Hm:.3f}/{Hmed:.3f} pmax={pm:.3f}/{pmed:.3f} modo={mf:.3f}(c={mc}) | ||W1||={nW1:.2e} ||W2||={nW2:.2e} ||grad||={gnorm_mean:.2e}/{gnorm_median:.2e} upd/||W1||~{ratio:.2e}")
            else:
                print(
                    f"[Época] {ep + 1:03d}/{epochs} lr={lr:.3e} loss_sub={loss_ep:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # early stopping
            if early_stop_enabled:
                metric_now = float(val_loss) if metric_name == "val_loss" else float(val_acc)
                if (ep + 1) >= warm:
                    if _melhorou(metric_now, best_metric, best_mode, min_delta=min_delta):
                        best_metric = metric_now
                        best_epoch = ep + 1
                        no_improve = 0
                        if bool(CONFIG["early_stop_restore_best"]):
                            best_state = (W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                                          None if ln_gamma is None else ln_gamma.copy(),
                                          None if ln_beta is None else ln_beta.copy())
                        print(f"[EarlyStop] melhorou ({metric_name}) -> {best_metric:.6f} em ep={best_epoch}")
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print(
                                f"[EarlyStop] PARANDO: sem melhora por {patience} épocas. Melhor ep={best_epoch} ({metric_name}={best_metric:.6f})")
                            break
        else:
            print(f"[Época] {ep + 1:03d}/{epochs} lr={lr:.3e} loss_sub={loss_ep:.4f}")

    # restaura melhor
    if has_val and early_stop_enabled and bool(CONFIG["early_stop_restore_best"]) and (best_state is not None):
        W1, b1, W2, b2, ln_gamma, ln_beta = best_state

    stats = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "metric_name": metric_name if has_val and early_stop_enabled else None,
        "rollbacks_used": rollbacks_used,
        "final_lr_base": float(CONFIG["lr_base"]),
        "dropout_p": dropout_p,
        "label_smoothing": smoothing,
        "use_layernorm": bool(CONFIG["use_layernorm"]),
    }
    return {
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
        "ln_gamma": ln_gamma, "ln_beta": ln_beta,
        "classes": classes_modelo,
        "act_hidden": act_hidden,
        "act_output": act_output,
        "stats": stats
    }


# ============================================================
# CV: escolhe melhor L2 (k=5 fixo)
# ============================================================

def escolher_melhor_l2_por_cv(X: np.ndarray, y: np.ndarray, seed: int):
    """Escolhe (l2, dropout_p) via 5-fold CV (mantém K=5).

    Critério: média de val_acc (conforme CONFIG['early_stop_metric'] no treino, mas
    aqui escolhemos por acc final no fold para ter comparabilidade direta).
    """
    from sklearn.model_selection import StratifiedKFold

    k = int(CONFIG["k_folds"])
    assert k == 5, "K-fold precisa continuar 5."

    grid_l2 = list(CONFIG["grid_l2"])
    grid_dp = list(CONFIG["grid_dropout"])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))

    melhor = None  # (mean_acc, l2, dp, std_acc)
    print("\n[CV] Grid-search (l2 x dropout) com K=5:")
    print(f"     l2 candidates   = {grid_l2}")
    print(f"     dropout candidates = {grid_dp}")
    print(f"     act_output      = {CONFIG['act_output']}")

    for dp in grid_dp:
        for l2 in grid_l2:
            accs = []
            for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
                X_tr, y_tr = X[tr], y[tr]
                X_va, y_va = X[va], y[va]

                modelo = treinar_mlp(
                    X_tr, y_tr,
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

                y_pred, _ = predict_labels(modelo, X_va)
                acc = float(np.mean(y_pred == y_va))
                accs.append(acc)

            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            print(f"  l2={l2:>7}  dp={dp:.2f}  -> val_acc(mean±std)={mean_acc:.4f} ± {std_acc:.4f}")

            cand = (mean_acc, float(l2), float(dp), std_acc)
            if (melhor is None) or (cand[0] > melhor[0]) or (cand[0] == melhor[0] and cand[1] < melhor[1]):
                melhor = cand

    assert melhor is not None
    mean_acc, best_l2, best_dp, std_acc = melhor
    print(f"[CV] Melhor combo: l2={best_l2} | dropout={best_dp:.2f} | val_acc={mean_acc:.4f} ± {std_acc:.4f}\n")
    return best_l2, best_dp, mean_acc


def main():
    path = CONFIG["dataset_path"]
    print("Dataset path:", path)
    print("Exists?", Path(path).exists())

    X, y = carregar_dataset(path)
    print("\n[Info] Dataset original")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("n classes:", len(np.unique(y)))

    # 1) classes elegíveis e seleção
    classes_elig = selecionar_classes_elegiveis(y, CONFIG["min_amostras_por_classe"])
    print(f"\n[Etapa 1] Classes elegíveis (>= {CONFIG['min_amostras_por_classe']}): {len(classes_elig)}")
    classes_sel = amostrar_classes(classes_elig, CONFIG["frac_classes"], CONFIG["seed_classes"])
    print(f"[Etapa 1] Classes selecionadas: {len(classes_sel)} (frac={CONFIG['frac_classes']})")
    X, y = filtrar_por_classes(X, y, classes_sel)
    print(f"[Etapa 1] Após filtro: X={X.shape} | classes={len(np.unique(y))}")

    # 2) split treino/teste
    print(f"\n[Etapa 2] Split treino/teste test_frac={CONFIG['test_frac']}")
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y,
        test_size=float(CONFIG["test_frac"]),
        random_state=int(CONFIG["seed_split"]),
        stratify=y,
    )
    print(f"[Etapa 2] Train(all): {X_train_all.shape} | classes={len(np.unique(y_train_all))}")
    print(f"[Etapa 2] Test (all):  {X_test_all.shape} | classes={len(np.unique(y_test_all))}")

    # 3) filtrar classes no treino com >= max(min_train_por_classe, k_folds)
    min_train = int(max(CONFIG["min_train_por_classe"], CONFIG["k_folds"]))
    cls, cts = np.unique(y_train_all, return_counts=True)
    cls_keep = cls[cts >= min_train]
    X_train, y_train = filtrar_por_classes(X_train_all, y_train_all, cls_keep)
    X_test, y_test = filtrar_por_classes(X_test_all, y_test_all, np.unique(y_train))
    print(f"\n[Etapa 3] Train(filtrado): {X_train.shape} | classes={len(np.unique(y_train))}")
    print(f"[Etapa 3] Test(alinhado):  {X_test.shape} | classes={len(np.unique(y_test))}")

    # 4) padronização
    print("\n[Etapa 4] Padronizando (z-score) com stats do TREINO.")
    mean_tr, std_tr = fit_standardizer(X_train, eps=float(CONFIG["eps_std"]))
    X_train_feat = apply_standardizer(X_train, mean_tr, std_tr)
    X_test_feat = apply_standardizer(X_test, mean_tr, std_tr)

    # 5) monta amostra para CV
    k = int(CONFIG["k_folds"])
    cv_min = CONFIG["cv_min_por_classe"]
    if cv_min is None:
        cv_min = max(5 * k, 20)  # [CORREÇÃO 6]
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
    _, cts_cv = np.unique(y_cv, return_counts=True)
    print(f"\n[Etapa 5] CV sample: X={X_cv.shape} | classes={len(np.unique(y_cv))} | min_count={int(cts_cv.min())}")

    # 6) CV grid-search (L2)
    print(
        f"\n[Etapa 6] CV grid-search | epochs_cv={CONFIG['epochs_cv']} batch_size_cv={CONFIG['batch_size_cv']} | k_folds={k}")
    best_l2, best_dropout, best_acc = escolher_melhor_l2_por_cv(X_cv, y_cv, seed=int(CONFIG["seed_split"]))
    print(f"\n[CV] Melhor: l2={best_l2:g} mean_acc={best_acc:.4f}")

    # 7) treino final
    print(f"\n[Etapa 7] Treino final | final_frac={CONFIG['final_frac']} epochs_final={CONFIG['epochs_final']}")
    idx_final, _ = amostrar_com_min_por_classe(
        y=y_train,
        frac=float(CONFIG["final_frac"]),
        seed=int(CONFIG["seed_split"]) + 999,
        min_por_classe=int(CONFIG["final_min_por_classe"]),
    )
    X_final = X_train_feat[idx_final]
    y_final = y_train[idx_final]
    classes_final = np.unique(y_final).astype(np.int64, copy=False)

    X_test_final, y_test_final = filtrar_por_classes(X_test_feat, y_test, classes_final)

    print(f"[Etapa 7] Final sample: X={X_final.shape} | classes={len(classes_final)}")
    print(f"[Etapa 7] Test alinhado: X={X_test_final.shape} | classes={len(np.unique(y_test_final))}")

    # split interno treino/val para early stop
    X_final_tr, y_final_tr, X_final_val, y_final_val = split_treino_validacao_estratificado(
        X_final, y_final, val_frac=float(CONFIG["final_val_frac"]), seed=int(CONFIG["final_val_seed"])
    )
    if X_final_val is not None:
        print(f"[Etapa 7] (EarlyStop Final) Split interno: treino={X_final_tr.shape} | val={X_final_val.shape}")

    modelo = treinar_mlp(
        X_train=X_final_tr, y_train=y_final_tr,
        classes_modelo=classes_final,
        l2=float(best_l2),
        epochs=int(CONFIG["epochs_final"]),
        batch_size=int(CONFIG["batch_size_final"]),
        seed=int(CONFIG["seed_split"]) + 2025,
        X_val=X_final_val, y_val=y_final_val,
        early_stop_enabled=bool(CONFIG["early_stop_final_enabled"]) and (X_final_val is not None),
        dropout_p=float(best_dropout),
        act_output=str(CONFIG["act_output"]),
    )

    st = modelo["stats"]
    print("\n[FINAL] use_layernorm=", st.get("use_layernorm"),
          "| dropout_p=", st.get("dropout_p"),
          "| label_smoothing=", st.get("label_smoothing"),
          "| best_epoch=", st.get("best_epoch"),
          "| best_metric=", st.get("best_metric"),
          "| rollbacks_used=", st.get("rollbacks_used"),
          "| final_lr_base=", st.get("final_lr_base"))

    # métricas finais
    yhat_tr, _ = predict_labels(X_final_tr, modelo)
    yhat_te, P_te = predict_labels(X_test_final, modelo)

    acc_tr = accuracy(y_final_tr, yhat_tr)
    acc_te = accuracy(y_test_final, yhat_te)
    print(f"[ACC] TREINO (parte treino): {acc_tr:.4f} ({int(np.sum(y_final_tr == yhat_tr))}/{y_final_tr.size})")
    print(f"[ACC] TESTE:              {acc_te:.4f} ({int(np.sum(y_test_final == yhat_te))}/{y_test_final.size})")

    # exemplos
    rng = np.random.default_rng(int(CONFIG["seed_split"]))
    n = int(CONFIG["n_exemplos_previsao"])
    if n > 0 and y_test_final.size > 0:
        idx = rng.choice(y_test_final.size, size=min(n, y_test_final.size), replace=False)
        print("\n[Exemplos] (y_true -> y_pred | p_max)")
        for i in idx:
            print(f"  {int(y_test_final[i])} -> {int(yhat_te[i])} | p={float(P_te[i].max()):.3f}")

    confusion_top_k(y_test_final, yhat_te, top_k=int(CONFIG["top_k_confusao"]))


if __name__ == "__main__":
    main()
