# -*- coding: utf-8 -*-
""""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Código de avaliação de modelo com tarefas de identificação e autenticação
Consome parâmetros de modelo treinado
Multilayer Perceptron com 1 Hidden Layer
"""

from __future__ import annotations

import sys
import math
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG DO SCRIPT (ajuste aqui)
# ============================================================

#Referências (em comum com avaliação para regressão logística)
# C. M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.
# K. P. Murphy, Machine Learning: A Probabilistic Perspective. MIT Press, 2012.
# T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006.
# C. J. van Rijsbergen, Information Retrieval, 2nd ed. Butterworth-Heinemann, 1979. (F-measure)
# D. M. W. Powers, “Evaluation: From Precision, Recall and F-Measure to ROC…,” Journal of Machine Learning Technologies, 2011.
# B. Schölkopf and A. J. Smola, Learning with Kernels. MIT Press, 2002.
# G. B. Huang et al., “Labeled Faces in the Wild…,” UMass Amherst, Tech. Rep. 07-49, 2007.
# F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet…,” in Proc. IEEE CVPR, 2015.
# J. Deng et al., “ArcFace…,” in Proc. IEEE CVPR, 2019.
# H. He and E. A. Garcia, “Learning from Imbalanced Data,” IEEE TKDE, vol. 21, no. 9, pp. 1263–1284, 2009.
# M. Sokolova and G. Lapalme, “A systematic analysis of performance measures…,” Information Processing & Management, 2009.
# P. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” in Proc. IJCAI, 1995.
# S. Kaufman, S. Rosset, and C. Perlich, “Leakage in Data Mining: Formulation, Detection, and Avoidance,” in Proc. ACM KDD, 2012.

#Referências (específicas para MLP)
# E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” Nature, vol. 323, pp. 533–536, 1986.



SCRIPT_CONFIG = {
    # nome do arquivo salvo pelo script de treino (na mesma pasta)
    "model_payload_file": "mlp_pm_model_and_classes.joblib",

    # se quiser sobrescrever o caminho do joblib HOG/dataset
    "dataset_path_override": None,  # ex.: r"C:\\...\\celeba_hog_128x128_o9.joblib"

    # aleatoriedade local deste script (não altera o split do treino, só prints/amostragens)
    "seed": 123,

    # identificação
    "one_vs_all_n_classes": 10,

    "roc": {
        "enable": True,
        "save_png": True,
        "show_plots": True,
        "png_name": "roc_ova_10classes.png",
    },

    # OBS: "macro_auc" abaixo mede AUC de IDENTIFICAÇÃO (one-vs-rest por classe).
    # Isso NÃO é AUC de autenticação (same vs different).
    "macro_auc": {
        "enable": False,
        "use_fast_auc": True,  # True: usa roc_auc_fast (mais rápido); False: usa roc_curve_simple
        "progress_every": 50,    # imprime status a cada N classes
    },

    # autenticação
    "auth": {
        "tune_metric": "f1",  # "f1" | "balanced_acc" | "acc"

        # tuning do threshold (amostragem):
        "pos_pairs_per_class": 50,  # tenta gerar até isso por classe (se houver amostras)
        "neg_pairs_total": 20000,  # negativos totais para tuning
        "threshold_grid_q": 401,  # quantis para varrer threshold (>=101 recomendado)

        # avaliação em pares:
        # modo "auto": se N for pequeno faz "full"; caso contrário faz "sample"
        "eval_mode": "auto",  # "auto" | "full" | "sample"
        "full_if_n_leq": 2500,  # se N <= isso, tenta full (O(N^2))
        "sample_pairs_if_large": 300000,  # #pares amostrados se N grande
        "pairs_fraction": 1.0,  # 0..1: fração dos pares amostrados usada em avaliação/AUC (1.0 = 100%)

        # matrizes por âncora
        "n_anchor_images": 10,

        # =====================================================
        # Macro AUC (AUTENTICAÇÃO) por identidade
        # =====================================================
        # O AUC global acima (same vs different) é um "micro-AUC":
        # cada PAR conta igualmente, então classes com mais imagens pesam mais.
        # Este bloco calcula um "macro-AUC":
        #   1) para cada identidade/classe c, amostra pares POS (c vs c) e NEG (c vs ~c)
        #   2) calcula AUC(c)
        #   3) tira a média simples dos AUC(c) em todas as classes
        "macro_auc": {
            "enable": True,
            "pos_pairs_per_class": 30,
            "neg_pairs_per_class": 60,
            "classes_fraction": 1.0,  # 1.0 = usa todas as classes; <1.0 amostra subconjunto
            "max_classes": 0,         # 0 = sem limite; >0 limita #classes (amostradas) por custo
            "progress_every": 100,
        },
    },

    # modo interativo (input)
    "enable_interactive_queries": False,
}


# ============================================================
# IO DO DATASET (HOG joblib)
# ============================================================

def carregar_dataset_joblib(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Aceita:
      - dict com chaves 'X' e 'y' (e possivelmente paths/meta)
      - tupla/lista (X, y)
    Retorna (X, y, meta)
    """
    obj = joblib.load(path)
    meta: Dict[str, Any] = {}

    if isinstance(obj, dict) and "X" in obj and "y" in obj:
        X = obj["X"]
        y = obj["y"]

        # tenta achar paths/ids
        for k in ("paths", "img_paths", "image_paths", "filenames", "files", "imgs", "img_files"):
            if k in obj:
                meta["paths"] = obj[k]
                break
        for k in ("ids", "image_ids", "img_ids"):
            if k in obj:
                meta["ids"] = obj[k]
                break

    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, y = obj

    else:
        raise ValueError("Formato do joblib do dataset não reconhecido. Esperado dict{'X','y'} ou tuple(X,y).")

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, meta


def selecionar_classes_elegiveis(y: np.ndarray, min_amostras: int) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts >= int(min_amostras)].astype(np.int64, copy=False)


def amostrar_classes(classes: np.ndarray, frac: float, seed: int) -> np.ndarray:
    frac = float(frac)
    classes = np.asarray(classes, dtype=np.int64)
    if frac >= 0.999999:
        return np.array(classes, copy=True)
    rng = np.random.default_rng(int(seed))
    n = len(classes)
    k = max(1, int(np.ceil(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    return np.sort(classes[idx]).astype(np.int64, copy=False)


def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_permitidas: np.ndarray,
                        paths: Optional[np.ndarray] = None,
                        ids: Optional[np.ndarray] = None,
                        idx_global: Optional[np.ndarray] = None):
    mask = np.isin(y, classes_permitidas)
    Xf = X[mask]
    yf = y[mask]
    pf = paths[mask] if paths is not None else None
    idf = ids[mask] if ids is not None else None
    igf = idx_global[mask] if idx_global is not None else None
    return Xf, yf, pf, idf, igf


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xf = X.astype(np.float32, copy=False)
    return ((Xf - mean) / std).astype(np.float32, copy=False)


# ============================================================
# MLP: Funções mínimas para inferência
# ============================================================

def stable_softmax(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    Zm = Z - Z.max(axis=1, keepdims=True)
    np.exp(Zm, out=Zm)
    Zm /= Zm.sum(axis=1, keepdims=True)
    return Zm


def _row_norm_forward(A: np.ndarray, eps: float):
    A = A.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(A * A, axis=1, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return A * inv, inv.astype(np.float32, copy=False)


def _col_norm_forward(W: np.ndarray, eps: float):
    W = W.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(W * W, axis=0, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return W * inv, inv.astype(np.float32, copy=False)


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
        return Z2

    if act_output == "cosine_softmax":
        A_hat, _ = _row_norm_forward(A1, eps=eps)
        W_hat, _ = _col_norm_forward(W2, eps=eps)
        Z2 = float(scale) * (A_hat @ W_hat)
        if bool(use_bias):
            Z2 = Z2 + b2
        return Z2

    raise ValueError(f"act_output desconhecida: {act_output}")


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


def layernorm_forward(Z: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
    Z = Z.astype(np.float32, copy=False)
    mu = Z.mean(axis=1, keepdims=True)
    var = Z.var(axis=1, keepdims=True)
    invstd = 1.0 / np.sqrt(var + np.float32(eps))
    xhat = (Z - mu) * invstd
    out = xhat * gamma + beta
    return out.astype(np.float32, copy=False)


def mlp_forward_inference(
        X: np.ndarray,
        modelo: Dict[str, Any],
        inference_params: Dict[str, Any],
):
    """
    Retorna (P, A1_pre) onde:
      - P: probabilidades (B,K)
      - A1_pre: ativação da camada escondida (B,H) (antes de dropout)
    """
    X = X.astype(np.float32, copy=False)

    W1 = modelo["W1"];
    b1 = modelo["b1"]
    W2 = modelo["W2"];
    b2 = modelo["b2"]
    act_hidden = modelo.get("act_hidden", inference_params.get("act_hidden", "relu"))
    act_output = modelo.get("act_output", inference_params.get("act_output", "cosine_softmax"))

    ln_gamma = modelo.get("ln_gamma", None)
    ln_beta = modelo.get("ln_beta", None)
    use_layernorm = bool(inference_params.get("use_layernorm", ln_gamma is not None and ln_beta is not None))
    ln_eps = float(inference_params.get("layernorm_eps", 1e-5))

    # camada 1
    Z1 = X @ W1 + b1
    if use_layernorm and (ln_gamma is not None) and (ln_beta is not None):
        Z1 = layernorm_forward(Z1, ln_gamma, ln_beta, eps=ln_eps)

    A1_pre = activation_forward(Z1, act_hidden)

    # logits e softmax
    Z2 = output_logits_forward(
        A1_pre, W2, b2,
        act_output=act_output,
        scale=float(inference_params.get("cosine_softmax_scale", 20.0)),
        eps=float(inference_params.get("cosine_softmax_eps", 1e-8)),
        use_bias=bool(inference_params.get("cosine_softmax_use_bias", False)),
    )
    P = stable_softmax(Z2)
    return P, A1_pre


def predict_labels(X: np.ndarray, classes: np.ndarray, modelo: Dict[str, Any], inference_params: Dict[str, Any]):
    P, _ = mlp_forward_inference(X, modelo, inference_params)
    idx = np.argmax(P, axis=1).astype(np.int64, copy=False)
    return classes[idx], P


def extract_embeddings(X: np.ndarray, modelo: Dict[str, Any], inference_params: Dict[str, Any]) -> np.ndarray:
    """
    Embedding = ativação da camada escondida (A1_pre) L2-normalizada (cosine-ready).
    """
    _, A1_pre = mlp_forward_inference(X, modelo, inference_params)
    A_hat, _ = _row_norm_forward(A1_pre, eps=float(inference_params.get("cosine_softmax_eps", 1e-8)))
    return A_hat.astype(np.float32, copy=False)


# ============================================================
# Métricas / Confusões
# ============================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_one_vs_all(y_true: np.ndarray, y_pred: np.ndarray, pos_class: int) -> Dict[str, int]:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    tpos = (y_true == int(pos_class))
    ppos = (y_pred == int(pos_class))

    TP = int(np.sum(tpos & ppos))
    FP = int(np.sum((~tpos) & ppos))
    FN = int(np.sum(tpos & (~ppos)))
    TN = int(np.sum((~tpos) & (~ppos)))
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


def print_confusion_binary(title: str, cm: Dict[str, int]):
    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    print(f"\n{title}")
    print("            Pred=0    Pred=1")
    print(f"True=0      {TN:6d}    {FP:6d}")
    print(f"True=1      {FN:6d}    {TP:6d}")
    denom = TP + TN + FP + FN
    acc = (TP + TN) / denom if denom > 0 else 0.0
    print(f"acc={acc:.4f} | TP={TP} FP={FP} FN={FN} TN={TN}")


# ============================================================
# Autenticação: threshold tuning + avaliação de pares
# ============================================================

def _sample_positive_pairs_per_class(y: np.ndarray, rng: np.random.Generator, per_class: int):
    y = np.asarray(y, dtype=np.int64)
    pairs = []

    classes = np.unique(y)
    for c in classes:
        idx = np.flatnonzero(y == c)
        m = idx.size
        if m < 2:
            continue
        max_pairs = m * (m - 1) // 2
        k = int(min(per_class, max_pairs))
        if k <= 0:
            continue

        got = 0
        tries = 0
        while got < k and tries < 10 * k:
            i = int(rng.choice(idx))
            j = int(rng.choice(idx))
            if i == j:
                tries += 1
                continue
            if i > j:
                i, j = j, i
            pairs.append((i, j, 1))
            got += 1
            tries += 1
    return pairs


def _sample_negative_pairs(y: np.ndarray, rng: np.random.Generator, total: int):
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    pairs = []
    if n < 2 or total <= 0:
        return pairs

    tries = 0
    got = 0
    while got < total and tries < 20 * total:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            tries += 1
            continue
        if y[i] == y[j]:
            tries += 1
            continue
        if i > j:
            i, j = j, i
        pairs.append((i, j, 0))
        got += 1
        tries += 1
    return pairs


def _sample_pairs_uniform(n: int, rng: np.random.Generator, n_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Amostra n_pairs pares (i<j) aproximadamente uniformes (com reposição).

    Observações:
      - Para n grande, amostragem SEM reposição/sem duplicatas é cara e desnecessária aqui.
      - Duplicatas têm efeito desprezível nas métricas/ROC quando n_pairs é grande.

    Retorna:
      ii, jj: arrays int64 com shape (n_pairs,), com ii[k] < jj[k] e ii[k] != jj[k].
    """
    n = int(n)
    n_pairs = int(n_pairs)
    if n < 2 or n_pairs <= 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64))

    ii = rng.integers(0, n, size=n_pairs, dtype=np.int64)
    jj = rng.integers(0, n, size=n_pairs, dtype=np.int64)

    # garante i != j (re-amostra apenas onde necessário)
    mask = (ii == jj)
    while np.any(mask):
        jj[mask] = rng.integers(0, n, size=int(np.sum(mask)), dtype=np.int64)
        mask = (ii == jj)

    # normaliza para i < j
    swap = ii > jj
    if np.any(swap):
        tmp = ii[swap].copy()
        ii[swap] = jj[swap]
        jj[swap] = tmp

    return ii, jj


def roc_auc_fast(scores: np.ndarray, truth: np.ndarray) -> float:
    """Calcula AUC ROC binária em O(n log n).

    Motivação: a função roc_curve_simple() existente enumera TODOS os thresholds únicos,
    o que explode para centenas de milhares/milhões de pares. Aqui usamos o método padrão:
    ordena por score decrescente e calcula cumulativos (equivalente ao sklearn).

    Ref (intuição/ROC): T. Fawcett, "An introduction to ROC analysis", Pattern Recognition Letters, 2006.
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    truth = np.asarray(truth).astype(bool).ravel()

    if scores.size == 0:
        return 0.0

    # total de positivos/negativos
    P = int(np.sum(truth))
    N = int(truth.size - P)
    if P == 0 or N == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    scores_s = scores[order]
    truth_s = truth[order].astype(np.int8)

    tp = np.cumsum(truth_s, dtype=np.int64)
    fp = np.cumsum(1 - truth_s, dtype=np.int64)

    # pontos apenas onde o score muda (thresholds distintos)
    distinct = np.where(np.diff(scores_s) != 0)[0]
    idx = np.r_[distinct, truth_s.size - 1]

    tpr = tp[idx] / float(P)
    fpr = fp[idx] / float(N)

    # inclui origem (0,0)
    tpr = np.r_[0.0, tpr.astype(np.float64, copy=False)]
    fpr = np.r_[0.0, fpr.astype(np.float64, copy=False)]

    # integral trapezoidal
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float(np.trapz(tpr, fpr))
    return auc


def auth_macro_auc_per_identity(
        emb: np.ndarray,
        P: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pmax: np.ndarray,
        rng: np.random.Generator,
        cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Macro ROC-AUC de AUTENTICAÇÃO por identidade (one-vs-all).

    Para cada classe/identidade c:
      - Positivos: pares (i,j) com y[i]==c e y[j]==c (mesma identidade)
      - Negativos: pares (i,j) com y[i]==c e y[j]!=c (ou vice-versa; aqui usamos i em c e j fora)
      - Calcula AUC(c) para diferentes scores de verificação:
          1) cosine  : dot de embeddings L2-normalizados (cosine similarity)
          2) prob_dot: dot entre vetores de probas por classe
          3) id+conf : (mesmo ID predito) * min(conf_i, conf_j)
    No fim, retorna média simples (macro) dos AUC(c) nas classes válidas.

    Observação importante:
      - Isto difere do AUC global (micro-AUC) que usa pares amostrados no dataset todo.
      - Macro-AUC dá o mesmo peso para cada identidade/classe, mesmo se algumas têm muito mais imagens.
    """
    enable = bool(cfg.get("enable", False))
    if not enable:
        return {"enabled": False}

    pos_pairs_per_class = int(cfg.get("pos_pairs_per_class", 30))
    neg_pairs_per_class = int(cfg.get("neg_pairs_per_class", 60))
    classes_fraction = float(cfg.get("classes_fraction", 1.0))
    max_classes = int(cfg.get("max_classes", 0))
    progress_every = int(cfg.get("progress_every", 100))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pmax = np.asarray(pmax, dtype=np.float32).ravel()
    emb = np.asarray(emb, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)

    classes = np.unique(y_true)
    if classes.size == 0:
        return {"enabled": True, "n_classes_used": 0}

    # amostra subconjunto de classes, se configurado
    if 0.0 < classes_fraction < 1.0:
        k = max(1, int(round(classes.size * classes_fraction)))
        classes = rng.choice(classes, size=k, replace=False)

    if max_classes > 0 and classes.size > max_classes:
        classes = rng.choice(classes, size=max_classes, replace=False)

    auc_cos_list: list[float] = []
    auc_prob_list: list[float] = []
    auc_conf_list: list[float] = []
    used_classes: list[int] = []

    t0 = time.time()
    for t, c in enumerate(classes.tolist(), start=1):
        idx = np.where(y_true == c)[0].astype(np.int64, copy=False)
        if idx.size < 2:
            continue
        idx_other = np.where(y_true != c)[0].astype(np.int64, copy=False)
        if idx_other.size < 1:
            continue

        k_pos = min(pos_pairs_per_class, 10_000_000)  # guard rail
        k_neg = min(neg_pairs_per_class, 10_000_000)
        if k_pos <= 0 or k_neg <= 0:
            continue

        # --- amostra positivos (c vs c)
        ii_pos = idx[rng.integers(0, idx.size, size=k_pos, dtype=np.int64)]
        jj_pos = idx[rng.integers(0, idx.size, size=k_pos, dtype=np.int64)]
        mask = (ii_pos == jj_pos)
        # evita i==j (re-amostra apenas onde necessário)
        while np.any(mask):
            jj_pos[mask] = idx[rng.integers(0, idx.size, size=int(np.sum(mask)), dtype=np.int64)]
            mask = (ii_pos == jj_pos)

        # --- amostra negativos (c vs ~c)
        ii_neg = idx[rng.integers(0, idx.size, size=k_neg, dtype=np.int64)]
        jj_neg = idx_other[rng.integers(0, idx_other.size, size=k_neg, dtype=np.int64)]

        ii = np.concatenate([ii_pos, ii_neg], axis=0)
        jj = np.concatenate([jj_pos, jj_neg], axis=0)
        truth = np.concatenate([
            np.ones((k_pos,), dtype=np.int8),
            np.zeros((k_neg,), dtype=np.int8),
        ], axis=0).astype(bool)

        # scores contínuos
        s_cos = np.sum(emb[ii] * emb[jj], axis=1)
        s_prob = np.sum(P[ii] * P[jj], axis=1)
        same_id = (y_pred[ii] == y_pred[jj]).astype(np.float32)
        conf = np.minimum(pmax[ii], pmax[jj])
        s_conf = conf * same_id

        auc_cos_list.append(float(roc_auc_fast(s_cos, truth)))
        auc_prob_list.append(float(roc_auc_fast(s_prob, truth)))
        auc_conf_list.append(float(roc_auc_fast(s_conf, truth)))
        used_classes.append(int(c))

        if progress_every > 0 and (t % progress_every == 0):
            dt = max(1e-6, time.time() - t0)
            rate = t / dt
            remaining = (len(classes) - t) / max(1e-9, rate)
            print(f"  [AUTH][macro-AUC] classes processadas: {t}/{len(classes)} | usadas={len(used_classes)} | "
                  f"vel={rate:.1f} cls/s | ETA~{remaining:.1f}s")

    if len(used_classes) == 0:
        return {"enabled": True, "n_classes_used": 0}

    return {
        "enabled": True,
        "n_classes_used": int(len(used_classes)),
        "classes_used": used_classes,
        "cos_mean": float(np.mean(auc_cos_list)),
        "prob_mean": float(np.mean(auc_prob_list)),
        "conf_mean": float(np.mean(auc_conf_list)),
        "cos_median": float(np.median(auc_cos_list)),
        "prob_median": float(np.median(auc_prob_list)),
        "conf_median": float(np.median(auc_conf_list)),
    }


def _pair_dot_scores_batched(A: np.ndarray, ii: np.ndarray, jj: np.ndarray,
                            batch: int = 50000, label: str = "pairs") -> np.ndarray:
    """Calcula dot(A[ii], A[jj]) em batches com tracker de progresso.

    Usado para autenticação por similaridade (ex.: cosine) e para prob_dot (dot entre vetores de probas).
    """
    A = np.asarray(A, dtype=np.float32)
    ii = np.asarray(ii, dtype=np.int64).ravel()
    jj = np.asarray(jj, dtype=np.int64).ravel()
    n_pairs = int(ii.size)
    out = np.empty((n_pairs,), dtype=np.float32)

    if n_pairs == 0:
        return out

    # Ajuste dinâmico de batch para evitar estouro de memória quando a dimensão é grande
    # (ex.: P tem milhares de classes; fancy-indexing já materializa (batch, dim)).
    batch = int(max(1, batch))
    if A.ndim == 2:
        d = int(A.shape[1])
        max_elems = 25_000_000  # ~100MB por matriz float32
        if d > 0:
            batch = min(batch, max(1, max_elems // d))

    # tracker: imprime ~a cada 10% ou a cada 50k, o que ocorrer primeiro
    step_print = max(50000, (n_pairs // 10) if n_pairs >= 10 else 1)

    done = 0
    for start in range(0, n_pairs, batch):
        end = min(n_pairs, start + batch)
        Ai = A[ii[start:end]]
        Aj = A[jj[start:end]]
        out[start:end] = np.einsum("ij,ij->i", Ai, Aj, optimize=True).astype(np.float32, copy=False)

        done = end
        if done == n_pairs or (done % step_print) == 0:
            pct = 100.0 * done / float(n_pairs)
            print(f"    [AUC-pairs] {label}: {done}/{n_pairs} ({pct:.1f}%)")

    return out

    batch = int(max(1, batch))
    # tracker: imprime ~a cada 10% ou a cada 50k, o que ocorrer primeiro
    step_print = max(50000, (n_pairs // 10) if n_pairs >= 10 else 1)

    done = 0
    for start in range(0, n_pairs, batch):
        end = min(n_pairs, start + batch)
        out[start:end] = np.sum(A[ii[start:end]] * A[jj[start:end]], axis=1).astype(np.float32, copy=False)
        done = end
        if done == n_pairs or (done % step_print) == 0:
            pct = 100.0 * done / float(n_pairs)
            print(f"    [AUC-pairs] {label}: {done}/{n_pairs} ({pct:.1f}%)")
    return out


def _binary_counts_from_pred_truth(pred: np.ndarray, truth: np.ndarray) -> Dict[str, int]:
    pred = np.asarray(pred).astype(bool)
    truth = np.asarray(truth).astype(bool)
    TP = int(np.sum(pred & truth))
    TN = int(np.sum((~pred) & (~truth)))
    FP = int(np.sum(pred & (~truth)))
    FN = int(np.sum((~pred) & truth))
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def _binary_metrics_from_counts(cm: Dict[str, int]) -> Dict[str, float]:
    TP = float(cm["TP"]);
    TN = float(cm["TN"]);
    FP = float(cm["FP"]);
    FN = float(cm["FN"])
    denom = TP + TN + FP + FN
    acc = (TP + TN) / denom if denom > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    tpr = rec
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "balanced_acc": float(bal_acc)}


def _score_summary(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if scores.size == 0:
        return {"n": 0}
    qs = np.quantile(scores, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).astype(np.float32)
    return {
        "n": int(scores.size),
        "min": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "median": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def build_tuning_pairs(y: np.ndarray, rng: np.random.Generator,
                       pos_pairs_per_class: int, neg_pairs_total: int,
                       balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Gera pares (i,j) com rótulo binário de "mesma classe?" (tt=1) ou "classes diferentes?" (tt=0).
    Tuning balanceado: limita #negativos ~ #positivos para evitar "acurácia alta só chutando negativo".
    """
    y = np.asarray(y, dtype=np.int64)
    pos_pairs = _sample_positive_pairs_per_class(y, rng, per_class=int(pos_pairs_per_class))
    n_pos = int(len(pos_pairs))
    n_neg_target = int(neg_pairs_total)
    if balanced and n_pos > 0:
        n_neg_target = min(int(neg_pairs_total), n_pos)
    neg_pairs = _sample_negative_pairs(y, rng, total=int(n_neg_target))
    pairs = pos_pairs + neg_pairs
    if len(pairs) == 0:
        ii = np.zeros((0,), dtype=np.int64)
        jj = np.zeros((0,), dtype=np.int64)
        tt = np.zeros((0,), dtype=np.int8)
        info = {"pairs_used": 0, "pos_pairs": 0, "neg_pairs": 0}
        return ii, jj, tt, info

    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    tt = np.array([p[2] for p in pairs], dtype=np.int8)
    info = {"pairs_used": int(len(pairs)), "pos_pairs": int(len(pos_pairs)), "neg_pairs": int(len(neg_pairs))}
    return ii, jj, tt, info


def tune_threshold_scores(scores: np.ndarray, truth: np.ndarray, q_grid: int,
                          metric: str = "f1") -> Tuple[float, Dict[str, Any]]:
    """
    Escolhe threshold maximizando F1 ou balanced-acc (ou acc).
    - scores: valores contínuos (maior = mais provável ser a mesma classe)
    - truth: 1 (mesma classe) / 0 (diferente)
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    truth = np.asarray(truth, dtype=np.int8).ravel()
    if scores.size == 0:
        return 0.5, {"note": "sem scores para tuning; usando thr=0.5"}

    q_grid = int(max(21, q_grid))
    qs = np.linspace(0.0, 1.0, q_grid, dtype=np.float32)
    thrs = np.unique(np.quantile(scores, qs))
    best_thr = float(thrs[0])
    best_val = -1.0
    best_cm = None
    best_metrics = None

    metric = (metric or "f1").strip().lower()
    if metric not in ("f1", "balanced_acc", "acc"):
        metric = "f1"

    for thr in thrs:
        pred = scores >= float(thr)
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        mets = _binary_metrics_from_counts(cm)
        val = mets["f1"] if metric == "f1" else (mets["balanced_acc"] if metric == "balanced_acc" else mets["acc"])
        if val > best_val:
            best_val = float(val)
            best_thr = float(thr)
            best_cm = cm
            best_metrics = mets

    stats = {
        "metric_optimized": metric,
        "best_metric_value": float(best_val),
        "best_thr": float(best_thr),
        "best_cm": best_cm,
        "best_metrics": best_metrics,
        "thr_candidates": int(thrs.size),
        "scores_summary": _score_summary(scores),
        "pos_scores_summary": _score_summary(scores[truth == 1]),
        "neg_scores_summary": _score_summary(scores[truth == 0]),
    }
    return best_thr, stats


def tune_threshold_identity_confidence(y_pred: np.ndarray, pmax: np.ndarray,
                                       ii: np.ndarray, jj: np.ndarray, truth: np.ndarray,
                                       q_grid: int, metric: str = "f1") -> Tuple[float, Dict[str, Any]]:
    """
    Regra de verificação baseada em identidade prevista e confiança:
      same = (y_pred[i] == y_pred[j]) AND (min(pmax_i, pmax_j) >= thr_conf)

    Faz tuning do thr_conf (balanceado via truth fornecido).
    """
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pmax = np.asarray(pmax, dtype=np.float32)
    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int8).ravel()
    if ii.size == 0:
        return 0.5, {"note": "sem pares para tuning; usando thr=0.5"}

    same_id = (y_pred[ii] == y_pred[jj])
    conf = np.minimum(pmax[ii], pmax[jj]).astype(np.float32, copy=False)

    # thresholds em [0,1] por quantis do conf (mas inclui 0 e 1)
    q_grid = int(max(21, q_grid))
    qs = np.linspace(0.0, 1.0, q_grid, dtype=np.float32)
    thrs = np.unique(np.quantile(conf, qs))
    best_thr = float(thrs[0])
    best_val = -1.0
    best_cm = None
    best_metrics = None

    metric = (metric or "f1").strip().lower()
    if metric not in ("f1", "balanced_acc", "acc"):
        metric = "f1"

    for thr in thrs:
        pred = same_id & (conf >= float(thr))
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        mets = _binary_metrics_from_counts(cm)
        val = mets["f1"] if metric == "f1" else (mets["balanced_acc"] if metric == "balanced_acc" else mets["acc"])
        if val > best_val:
            best_val = float(val)
            best_thr = float(thr)
            best_cm = cm
            best_metrics = mets

    stats = {
        "metric_optimized": metric,
        "best_metric_value": float(best_val),
        "best_thr": float(best_thr),
        "best_cm": best_cm,
        "best_metrics": best_metrics,
        "thr_candidates": int(thrs.size),
        "conf_summary_all": _score_summary(conf),
        "conf_summary_sameid": _score_summary(conf[same_id]),
        "conf_summary_diffid": _score_summary(conf[~same_id]),
    }
    return best_thr, stats


def eval_auth_pairs_full_prob(P: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    """Avalia todas as combinações com score = dot(P_i, P_j)."""
    P = np.asarray(P, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        sims = P[i + 1:] @ P[i]  # (n-i-1,)
        pred = sims >= float(thr)
        truth = (y[i + 1:] == y[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    acc = (TP + TN) / pairs if pairs > 0 else 0.0
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            **{f"m_{k}": v for k, v in mets.items()}}


def eval_auth_pairs_full_identity_confidence(y_pred: np.ndarray, pmax: np.ndarray, y_true: np.ndarray,
                                             thr_conf: float) -> Dict[str, Any]:
    """Avalia todas as combinações com regra: same_id_pred & min_conf >= thr_conf."""
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pmax = np.asarray(pmax, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)
    n = int(y_true.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        same_id = (y_pred[i + 1:] == y_pred[i])
        conf = np.minimum(pmax[i + 1:], pmax[i])
        pred = same_id & (conf >= float(thr_conf))
        truth = (y_true[i + 1:] == y_true[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    acc = (TP + TN) / pairs if pairs > 0 else 0.0
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            **{f"m_{k}": v for k, v in mets.items()}}


def roc_curve_simple(scores: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ROC simples (sem sklearn). Retorna fpr, tpr, thresholds, auc.
    truth: 1/0
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    truth = np.asarray(truth, dtype=np.int8).ravel()
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0
    thrs = np.unique(scores)[::-1]
    # adiciona um thr > max para ponto (0,0)
    thrs = np.concatenate(([float(np.max(scores) + 1.0)], thrs, [float(np.min(scores) - 1.0)])).astype(np.float32)
    Ppos = float(np.sum(truth == 1))
    Nneg = float(np.sum(truth == 0))
    tpr = []
    fpr = []
    for thr in thrs:
        pred = scores >= float(thr)
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        TP = cm["TP"];
        FP = cm["FP"]
        tpr.append(TP / Ppos if Ppos > 0 else 0.0)
        fpr.append(FP / Nneg if Nneg > 0 else 0.0)
    fpr = np.asarray(fpr, dtype=np.float32)
    tpr = np.asarray(tpr, dtype=np.float32)
    # ordena por fpr crescente
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = float(np.trapezoid(tpr, fpr)) if (hasattr(np, "trapezoid") and fpr.size > 1) else (float(np.trapz(tpr, fpr)) if fpr.size > 1 else 0.0)
    return fpr, tpr, thrs.astype(np.float32), auc


def plot_roc_ova_for_classes(pick_classes: List[int], y_true: np.ndarray, proba: np.ndarray,
                             classes_modelo: np.ndarray, out_dir: Path, roc_cfg: Dict[str, Any]):
    """Plota ROC one-vs-all (score = probabilidade da classe)."""
    if not roc_cfg or not bool(roc_cfg.get("enable", True)):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ROC] Falha ao importar matplotlib ({e}). Pulando ROC.")
        return

    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float32)
    classes_modelo = np.asarray(classes_modelo, dtype=np.int64)
    pick_classes = [int(c) for c in pick_classes]

    plt.figure()
    any_curve = False
    for c in pick_classes:
        idx = np.where(classes_modelo == int(c))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        scores = proba[:, k]
        truth = (y_true == int(c)).astype(np.int8)
        fpr, tpr, _thrs, auc = roc_curve_simple(scores, truth)
        if fpr.size == 0:
            continue
        any_curve = True
        plt.plot(fpr, tpr, label=f"c={c} AUC={auc:.3f}")

    if not any_curve:
        print("[ROC] Sem curvas para plotar.")
        return

    plt.plot([0, 1], [0, 1], linestyle="--", label="aleatório")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC One-vs-All (10 classes)")
    plt.legend(loc="lower right")

    if bool(roc_cfg.get("save_png", True)):
        png_name = str(roc_cfg.get("png_name", "roc_ova_10classes.png"))
        out_path = out_dir / png_name
        try:
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[ROC] PNG salvo em: {out_path}")
        except Exception as e:
            print(f"[ROC] Falha ao salvar PNG ({e}).")

    if bool(roc_cfg.get("show_plots", True)):
        plt.show()
    else:
        plt.close()


def eval_auth_pairs_full(emb: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    """
    Avalia todas as combinações (i<j) sem armazenar índices enormes.
    Retorna contagens TP/TN/FP/FN e acc.
    """
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        sims = emb[i + 1:] @ emb[i]  # (n-i-1,)
        pred = sims >= float(thr)
        truth = (y[i + 1:] == y[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    acc = (TP + TN) / pairs if pairs > 0 else 0.0
    return {"mode": "full", "pairs": int(pairs), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def eval_auth_pairs_sample(emb: np.ndarray, y: np.ndarray, thr: float, rng: np.random.Generator, n_pairs: int):
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2 or n_pairs <= 0:
        return {"note": "N<2 ou n_pairs<=0", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    got = 0
    tries = 0
    while got < n_pairs and tries < 10 * n_pairs:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            tries += 1
            continue
        if i > j:
            i, j = j, i
        sim = float(np.dot(emb[i], emb[j]))
        pred = sim >= float(thr)
        truth = (y[i] == y[j])
        if pred and truth:
            TP += 1
        elif (not pred) and (not truth):
            TN += 1
        elif pred and (not truth):
            FP += 1
        else:
            FN += 1
        got += 1
        tries += 1

    acc = (TP + TN) / got if got > 0 else 0.0
    return {"mode": "sample", "pairs": int(got), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN}


# ============================================================
# Split / reconstrução de avaliação
# ============================================================

def _train_test_split_with_meta(X: np.ndarray, y: np.ndarray,
                                paths: Optional[np.ndarray], ids: Optional[np.ndarray],
                                test_size: float, seed: int):
    idx = np.arange(y.size, dtype=np.int64)
    idx_tr, idx_te = train_test_split(
        idx,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    def take(a, idx_):
        return a[idx_] if a is not None else None

    return (X[idx_tr], X[idx_te],
            y[idx_tr], y[idx_te],
            take(paths, idx_tr), take(paths, idx_te),
            take(ids, idx_tr), take(ids, idx_te))


def build_eval_split_from_payload(X: np.ndarray, y: np.ndarray,
                                  paths: Optional[np.ndarray],
                                  ids: Optional[np.ndarray],
                                  payload_conf: Dict[str, Any],
                                  standardizer: Dict[str, Any],
                                  classes_modelo: np.ndarray):
    idx_global = np.arange(y.size, dtype=np.int64)

    classes_elig = selecionar_classes_elegiveis(y, int(payload_conf["min_amostras_por_classe"]))
    classes_sel = amostrar_classes(classes_elig, float(payload_conf["frac_classes"]), int(payload_conf["seed_classes"]))
    X1, y1, p1, id1, _ = filtrar_por_classes(X, y, classes_sel, paths=paths, ids=ids, idx_global=idx_global)

    X_train_all, X_test_all, y_train_all, y_test_all, p_train_all, p_test_all, id_train_all, id_test_all = \
        _train_test_split_with_meta(
            X1, y1, p1, id1,
            test_size=float(payload_conf["test_frac"]),
            seed=int(payload_conf["seed_split"]),
        )

    min_train = int(max(int(payload_conf["min_train_por_classe"]), int(payload_conf["k_folds"])))
    cls, cts = np.unique(y_train_all, return_counts=True)
    cls_keep = cls[cts >= min_train]

    X_train, y_train, p_train, id_train, _ = filtrar_por_classes(X_train_all, y_train_all, cls_keep,
                                                                 paths=p_train_all, ids=id_train_all,
                                                                 idx_global=None)
    cls_train = np.unique(y_train).astype(np.int64, copy=False)
    X_test, y_test, p_test, id_test, _ = filtrar_por_classes(X_test_all, y_test_all, cls_train,
                                                             paths=p_test_all, ids=id_test_all,
                                                             idx_global=None)

    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    X_test_feat = apply_standardizer(X_test, mean, std)

    X_test_final, y_test_final, p_test_final, id_test_final, _ = filtrar_por_classes(
        X_test_feat, y_test, classes_modelo,
        paths=p_test, ids=id_test, idx_global=None
    )
    return X_test_final, y_test_final, p_test_final, id_test_final


def main():
    rng = np.random.default_rng(int(SCRIPT_CONFIG["seed"]))

    out_dir = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    payload_path = out_dir / str(SCRIPT_CONFIG["model_payload_file"])

    if not payload_path.exists():
        raise FileNotFoundError(f"Não achei o payload do modelo em: {payload_path}")

    payload = joblib.load(payload_path)
    if not isinstance(payload, dict) or "modelo" not in payload or "classes_usadas" not in payload:
        raise ValueError("Payload do modelo inválido. Esperado dict com chaves 'modelo' e 'classes_usadas'.")

    modelo = payload["modelo"]
    classes_modelo = np.asarray(payload["classes_usadas"], dtype=np.int64)
    standardizer = payload.get("standardizer", None)
    inference_params = payload.get("inference_params", {}) or {}
    conf = payload.get("config_snapshot", {}) or {}

    print("\n[LOAD] Payload do modelo carregado com sucesso!")
    print("  arquivo:", payload_path)
    print("  n_classes_modelo:", int(classes_modelo.size))

    n_show = min(20, classes_modelo.size)
    print("\n[INFO] 20 IDs de classes (exemplo):")
    print(" ", classes_modelo[:n_show].tolist())

    print("\n[INFO] Exemplos de parâmetros do modelo:")
    W1 = modelo["W1"];
    b1 = modelo["b1"];
    W2 = modelo["W2"];
    b2 = modelo["b2"]
    print(f"  W1: {tuple(W1.shape)} | b1: {tuple(b1.shape)}")
    print(f"  W2: {tuple(W2.shape)} | b2: {tuple(b2.shape)}")
    print(f"  act_hidden: {modelo.get('act_hidden', inference_params.get('act_hidden', '??'))}")
    print(f"  act_output: {modelo.get('act_output', inference_params.get('act_output', '??'))}")
    print(
        f"  use_layernorm: {inference_params.get('use_layernorm', None)} | layernorm_eps: {inference_params.get('layernorm_eps', None)}")
    print(
        f"  cosine_softmax_scale: {inference_params.get('cosine_softmax_scale', None)} | cosine_softmax_use_bias: {inference_params.get('cosine_softmax_use_bias', None)}")
    if "metrics" in payload:
        print("  metrics(payload):", payload["metrics"])
    if "best_hparams" in payload:
        print("  best_hparams(payload):", payload["best_hparams"])

    if standardizer is None or "mean" not in standardizer or "std" not in standardizer:
        raise ValueError("Payload não contém 'standardizer' com mean/std; não dá para padronizar em inferência.")

    dataset_path = SCRIPT_CONFIG["dataset_path_override"] or conf.get("dataset_path", None)
    if not dataset_path:
        raise ValueError("Não há dataset_path no payload e dataset_path_override=None. Ajuste SCRIPT_CONFIG.")
    dataset_path = str(dataset_path)

    print("\n[DATA] Dataset path:", dataset_path)
    print("[DATA] Exists?", Path(dataset_path).exists())

    X_raw, y_raw, meta = carregar_dataset_joblib(dataset_path)
    paths = meta.get("paths", None)
    ids = meta.get("ids", None)

    print("\n[DATA] Dataset carregado:")
    print("  X:", tuple(X_raw.shape), X_raw.dtype)
    print("  y:", tuple(y_raw.shape), y_raw.dtype)
    print("  n_classes_total:", int(np.unique(y_raw).size))
    print("  paths:", "OK" if paths is not None else "(não encontrado no joblib)")
    print("  ids:", "OK" if ids is not None else "(não encontrado no joblib)")

    X_eval, y_eval, paths_eval, ids_eval = build_eval_split_from_payload(
        X=X_raw, y=y_raw,
        paths=np.asarray(paths) if paths is not None else None,
        ids=np.asarray(ids) if ids is not None else None,
        payload_conf=conf,
        standardizer=standardizer,
        classes_modelo=classes_modelo,
    )

    print("\n[EVAL] Conjunto de avaliação (TESTE alinhado):")
    print("  X_eval:", tuple(X_eval.shape), X_eval.dtype)
    print("  y_eval:", tuple(y_eval.shape), y_eval.dtype)
    print("  classes_present:", int(np.unique(y_eval).size))
    if paths_eval is not None:
        print("  paths_eval:", tuple(paths_eval.shape))
    if ids_eval is not None:
        print("  ids_eval:", tuple(ids_eval.shape))

    print("\n[ID] IMPORTANTE: neste script, o 'ID' de amostra para consultas = índice em X_eval (0..N-1).")
    print(f"     N={int(y_eval.size)} (IDs válidos: 0..{int(y_eval.size) - 1})")
    # =========================
    # TUNING (TREINO) para autenticação (anti-leakage)
    # =========================
    print("\n[SPLIT] Reconstruindo conjunto de TUNING (treino alinhado ao modelo) para autenticação (anti-leakage)...")

    idx_global_tune = np.arange(y_raw.size, dtype=np.int64)

    classes_elig_tune = selecionar_classes_elegiveis(y_raw, int(conf["min_amostras_por_classe"]))
    classes_sel_tune = amostrar_classes(classes_elig_tune, float(conf["frac_classes"]), int(conf["seed_classes"]))

    X1_t, y1_t, p1_t, id1_t, _ = filtrar_por_classes(
        X_raw, y_raw, classes_sel_tune,
        paths=np.asarray(paths) if paths is not None else None,
        ids=np.asarray(ids) if ids is not None else None,
        idx_global=idx_global_tune
    )

    X_train_all_t, X_test_all_t, y_train_all_t, y_test_all_t, p_train_all_t, p_test_all_t, id_train_all_t, id_test_all_t = \
        _train_test_split_with_meta(
            X1_t, y1_t, p1_t, id1_t,
            test_size=float(conf["test_frac"]),
            seed=int(conf["seed_split"]),
        )

    min_train_t = int(max(int(conf["min_train_por_classe"]), int(conf["k_folds"])))
    cls_t, cts_t = np.unique(y_train_all_t, return_counts=True)
    cls_keep_t = cls_t[cts_t >= min_train_t]

    X_train_t, y_train_t, p_train_t, id_train_t, _ = filtrar_por_classes(
        X_train_all_t, y_train_all_t, cls_keep_t,
        paths=p_train_all_t, ids=id_train_all_t,
        idx_global=None
    )

    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    X_train_feat_t = apply_standardizer(X_train_t, mean, std)

    X_tune, y_tune, paths_tune, ids_tune, _ = filtrar_por_classes(
        X_train_feat_t, y_train_t, classes_modelo,
        paths=p_train_t, ids=id_train_t, idx_global=None
    )

    print("  X_tune:", tuple(X_tune.shape), X_tune.dtype)
    print("  y_tune:", tuple(y_tune.shape), y_tune.dtype, "| classes_present:", int(np.unique(y_tune).size))


    # =========================
    # IDENTIFICAÇÃO (multiclasse)
    # =========================
    # Objetivo: prever o ID (classe) de cada imagem.
    # - Entrada: X_eval (features HOG previamente extraídas).
    # - Saída do modelo: P (distribuição de probabilidade por classe) e y_pred (argmax).
    # - Métrica principal: acurácia top-1; complementos: confusão, e AUC macro (one-vs-rest).
    #
    # Referências (fundamentos):
    # - C. Bishop, *Pattern Recognition and Machine Learning* (classificação multiclasse).
    # - T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*.
    # - T. Fawcett, "An introduction to ROC analysis" (ROC/AUC), 2006.
    y_pred, P = predict_labels(X_eval, classes_modelo, modelo, inference_params)
    acc = accuracy(y_eval, y_pred)
    print("\n[IDENTIFICAÇÃO] Acurácia no conjunto de avaliação:")
    print(f"  acc={acc:.4f} ({int(np.sum(y_eval == y_pred))}/{int(y_eval.size)})")

    # one-vs-all
    n_ova = int(SCRIPT_CONFIG["one_vs_all_n_classes"])
    classes_present = np.unique(y_eval).astype(np.int64, copy=False)
    n_pick = min(n_ova, int(classes_present.size))
    pick = rng.choice(classes_present, size=n_pick, replace=False)

    print(f"\n[IDENTIFICAÇÃO] One-vs-all ({n_pick} classes aleatórias): classes={pick.tolist()}")
    for c in pick.tolist():
        cm = confusion_one_vs_all(y_eval, y_pred, int(c))
        print_confusion_binary(title=f"[One-vs-all] classe={int(c)}", cm=cm)

    # ROC (one-vs-all) para as mesmas 10 classes escolhidas acima (thresholds variados)
    plot_roc_ova_for_classes(
        pick_classes=pick.tolist(),
        y_true=y_eval,
        proba=P,
        classes_modelo=classes_modelo,
        out_dir=out_dir,
        roc_cfg=SCRIPT_CONFIG.get("roc", {}) or {}
    )

    # =========================
    # AUTENTICAÇÃO (verificação: same vs different)
    # =========================
    # Objetivo: dado um par de imagens, decidir se pertencem ao MESMO ID.
    # Estratégia: transformar cada imagem em uma representação e medir similaridade:
    # - cosine em embeddings (vetores contínuos do penúltimo layer / representação interna)
    # - prob_dot: dot entre distribuições P (afinidade entre posteriors)
    # - id+conf: regra discreta (mesmo y_pred) + confiança (min(pmax)) >= threshold
    #
    # Implementação:
    # 1) Extraímos embeddings e (opcionalmente) normalizamos para usar dot ≡ cosine.
    # 2) Fazemos tuning de thresholds em um conjunto separado (y_tune).
    # 3) Avaliamos em pares. Para N grande, amostramos pares para evitar O(N^2).
    # 4) Calculamos AUC ROC em pares com algoritmo O(n log n) (roc_auc_fast), pois
    #    enumerar todos os thresholds únicos não escala para centenas de milhares de pares.
    #
    # Referências (contexto de verificação por embeddings):
    # - Schroff et al., "FaceNet" (embeddings + cosine para verificação), 2015.
    # - Deng et al., "ArcFace" / Wang et al., "CosFace" (margens em espaço angular), 2018.
    # - T. Fawcett, 2006 (ROC/AUC).
    print("\n[AUTH] Extraindo embeddings...")
    emb = extract_embeddings(X_eval, modelo, inference_params)
    pmax = np.max(P, axis=1).astype(np.float32, copy=False)

    auth_cfg = SCRIPT_CONFIG.get("auth", {}) or {}
    tune_metric = str(auth_cfg.get("tune_metric", "f1")).strip().lower()
    if tune_metric not in ("f1", "balanced_acc", "acc"):
        tune_metric = "f1"

    print("[AUTH] Tuning balanceado de thresholds ...")
    # (anti-leakage) o tuning de thresholds deve usar TREINO (X_tune/y_tune), não o TESTE (X_eval/y_eval)
    y_pred_tune, P_tune = predict_labels(X_tune, classes_modelo, modelo, inference_params)
    emb_tune = extract_embeddings(X_tune, modelo, inference_params)
    pmax_tune = np.max(P_tune, axis=1).astype(np.float32, copy=False)

    ii_t, jj_t, tt_t, pair_info = build_tuning_pairs(
        y_tune, rng=rng,
        pos_pairs_per_class=int(auth_cfg.get("pos_pairs_per_class", 50)),
        neg_pairs_total=int(auth_cfg.get("neg_pairs_total", 20000)),
        balanced=True
    )

    if int(pair_info.get("pairs_used", 0)) == 0:
        print("[AUTH] Sem pares suficientes para tuning. Usando thresholds padrão.")
        thr_cos = 0.5
        thr_prob = 0.5
        thr_conf = 0.5
    else:
        sims_cos = np.sum(emb_tune[ii_t] * emb_tune[jj_t], axis=1).astype(np.float32, copy=False)
        sims_prob = np.sum(P_tune[ii_t] * P_tune[jj_t], axis=1).astype(np.float32, copy=False)

        pos_mask = (tt_t == 1)
        neg_mask = (tt_t == 0)
        print(f"  pares tuning: {pair_info} | métrica otimizada: {tune_metric}")

        # print sim_pos vs sim_neg
        print("\n  [SCORES] cosine: pos vs neg")
        print("    pos:", _score_summary(sims_cos[pos_mask]))
        print("    neg:", _score_summary(sims_cos[neg_mask]))
        print("\n  [SCORES] prob_dot: pos vs neg")
        print("    pos:", _score_summary(sims_prob[pos_mask]))
        print("    neg:", _score_summary(sims_prob[neg_mask]))

        thr_cos, stats_cos = tune_threshold_scores(
            sims_cos, tt_t, q_grid=int(auth_cfg.get("threshold_grid_q", 401)), metric=tune_metric
        )
        thr_prob, stats_prob = tune_threshold_scores(
            sims_prob, tt_t, q_grid=int(auth_cfg.get("threshold_grid_q", 401)), metric=tune_metric
        )
        thr_conf, stats_conf = tune_threshold_identity_confidence(
            y_pred=y_pred_tune, pmax=pmax_tune,
            ii=ii_t, jj=jj_t, truth=tt_t,
            q_grid=int(auth_cfg.get("threshold_grid_q", 401)), metric=tune_metric
        )

        print("\n  [TUNING] cosine:", {k: stats_cos.get(k) for k in (
        "metric_optimized", "best_metric_value", "best_thr", "best_cm", "best_metrics")})
        print("  [TUNING] prob_dot:", {k: stats_prob.get(k) for k in (
        "metric_optimized", "best_metric_value", "best_thr", "best_cm", "best_metrics")})
        print("  [TUNING] id+conf:", {k: stats_conf.get(k) for k in
                                      ("metric_optimized", "best_metric_value", "best_thr", "best_cm", "best_metrics")})

    eval_mode = str(auth_cfg.get("eval_mode", "auto")).lower()
    full_if_n_leq = int(auth_cfg.get("full_if_n_leq", 2500))
    n = int(y_eval.size)
    use_full = (eval_mode == "full") or (eval_mode == "auto" and n <= full_if_n_leq)
    print(f"\n[AUTH] Avaliação em pares: mode={'full' if use_full else 'sample'} | N={n}")

    if use_full:
        res_cos = eval_auth_pairs_full(emb, y_eval, thr=float(thr_cos))
        res_prob = eval_auth_pairs_full_prob(P, y_eval, thr=float(thr_prob))
        res_conf = eval_auth_pairs_full_identity_confidence(y_pred=y_pred, pmax=pmax, y_true=y_eval,
                                                            thr_conf=float(thr_conf))

        # AUC ROC em pares (amostrados) mesmo em modo "full"
        # Motivo: AUC exige scores contínuos; calcular ROC/AUC em TODOS os pares O(N^2) pode ficar pesado.
        # Aqui usamos a mesma amostragem controlada por `sample_pairs_if_large` * `pairs_fraction`.
        base_pairs = int(auth_cfg.get("sample_pairs_if_large", 300000))
        frac_pairs = float(auth_cfg.get("pairs_fraction", 1.0))
        frac_pairs = max(0.0, min(1.0, frac_pairs))
        n_pairs_auc = int(round(base_pairs * frac_pairs))
        if base_pairs > 0 and n_pairs_auc < 1:
            n_pairs_auc = 1

        if n_pairs_auc > 0:
            ii_auc, jj_auc = _sample_pairs_uniform(n=int(n), rng=rng, n_pairs=int(n_pairs_auc))
            truth_auc = (y_eval[ii_auc] == y_eval[jj_auc]).astype(np.int8)

            sims_cos_auc = _pair_dot_scores_batched(emb, ii_auc, jj_auc, batch=50000, label="cosine(AUC)")
            sims_prob_auc = _pair_dot_scores_batched(P, ii_auc, jj_auc, batch=50000, label="prob_dot(AUC)")

            same_id_auc = (y_pred[ii_auc] == y_pred[jj_auc])
            conf_auc = np.minimum(pmax[ii_auc], pmax[jj_auc]).astype(np.float32, copy=False)
            score_conf_auc = (conf_auc * same_id_auc.astype(np.float32)).astype(np.float32, copy=False)

            print("  [AUTH] ROC-AUC (autenticação): ordenando cosine(AUC)...")
            auc_cos = roc_auc_fast(sims_cos_auc, truth_auc == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando prob_dot(AUC)...")
            auc_prob = roc_auc_fast(sims_prob_auc, truth_auc == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando id+conf(AUC)...")
            auc_conf = roc_auc_fast(score_conf_auc, truth_auc == 1)

            print(f"  [AUTH] ROC-AUC (autenticação, same vs different, threshold-var; pares amostrados, n={int(ii_auc.size)}): "
                  f"cosine={auc_cos:.4f} | prob_dot={auc_prob:.4f} | id+conf={auc_conf:.4f}")
    else:
        # --- Amostragem de pares para N grande ---
        # Em vez de avaliar todos os pares O(N^2), amostramos uma quantidade configurável.
        # A nova flag `pairs_fraction` permite reduzir rapidamente o custo para prototipagem
        # (ex.: 0.2 usa 20% de `sample_pairs_if_large`).
        base_pairs = int(auth_cfg.get("sample_pairs_if_large", 300000))
        frac_pairs = float(auth_cfg.get("pairs_fraction", 1.0))
        frac_pairs = max(0.0, min(1.0, frac_pairs))
        n_pairs = int(round(base_pairs * frac_pairs))
        if base_pairs > 0 and n_pairs < 1:
            n_pairs = 1

        if n_pairs <= 0:
            res_cos = {"mode": "sample", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
            res_prob = {"mode": "sample", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
            res_conf = {"mode": "sample", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
        else:
            ii_s, jj_s = _sample_pairs_uniform(n=int(n), rng=rng, n_pairs=int(n_pairs))
            truth_s = (y_eval[ii_s] == y_eval[jj_s]).astype(np.int8)

            # cosine similarity (dot em embeddings L2-normalizados ≡ cosine)
            sims_cos = _pair_dot_scores_batched(emb, ii_s, jj_s, batch=50000, label="cosine")
            pred_cos = sims_cos >= float(thr_cos)
            cm_cos_s = _binary_counts_from_pred_truth(pred_cos, truth_s == 1)
            mets_cos_s = _binary_metrics_from_counts(cm_cos_s)
            res_cos = {"mode": "sample", "pairs": int(ii_s.size),
                       "TP": cm_cos_s["TP"], "TN": cm_cos_s["TN"], "FP": cm_cos_s["FP"], "FN": cm_cos_s["FN"],
                       **{f"m_{k}": v for k, v in mets_cos_s.items()}}

            # prob_dot: dot entre vetores de probas por classe (interpretação: "afinidade" de distribuição)
            sims_prob = _pair_dot_scores_batched(P, ii_s, jj_s, batch=50000, label="prob_dot")
            pred_prob = sims_prob >= float(thr_prob)
            cm_prob = _binary_counts_from_pred_truth(pred_prob, truth_s == 1)
            mets_prob = _binary_metrics_from_counts(cm_prob)
            res_prob = {"mode": "sample", "pairs": int(ii_s.size),
                        "TP": cm_prob["TP"], "TN": cm_prob["TN"], "FP": cm_prob["FP"], "FN": cm_prob["FN"],
                        **{f"m_{k}": v for k, v in mets_prob.items()}}

            # id+conf: mesmo ID predito E confiança conjunta >= thr_conf
            same_id = (y_pred[ii_s] == y_pred[jj_s])
            conf_s = np.minimum(pmax[ii_s], pmax[jj_s]).astype(np.float32, copy=False)
            pred_conf = same_id & (conf_s >= float(thr_conf))
            cm_conf = _binary_counts_from_pred_truth(pred_conf, truth_s == 1)
            mets_conf = _binary_metrics_from_counts(cm_conf)
            res_conf = {"mode": "sample", "pairs": int(ii_s.size),
                        "TP": cm_conf["TP"], "TN": cm_conf["TN"], "FP": cm_conf["FP"], "FN": cm_conf["FN"],
                        **{f"m_{k}": v for k, v in mets_conf.items()}}

            # AUC ROC em pares (com tracker de progresso no cálculo de scores acima)
            # Atenção: para ROC, usamos scores contínuos (não binarizados pelo threshold).
            score_conf = (conf_s * same_id.astype(np.float32)).astype(np.float32, copy=False)
            print("  [AUTH] ROC-AUC (autenticação): ordenando cosine...")
            auc_cos = roc_auc_fast(sims_cos, truth_s == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando prob_dot...")
            auc_prob = roc_auc_fast(sims_prob, truth_s == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando id+conf...")
            auc_conf = roc_auc_fast(score_conf, truth_s == 1)
            print(f"  [AUTH] ROC-AUC (autenticação, same vs different, threshold-var; pares amostrados, n={int(ii_s.size)}): "
                  f"cosine={auc_cos:.4f} | prob_dot={auc_prob:.4f} | id+conf={auc_conf:.4f}")

    # ------------------------------------------------------------
    # Macro AUC (AUTENTICAÇÃO) por identidade: média simples dos AUCs por classe
    # (equivalente ao seu "AUC composto" = mean(AUC one-vs-all) em autenticação)
    # ------------------------------------------------------------
    macro_auth_cfg = (auth_cfg.get("macro_auc", {}) or {})
    if bool(macro_auth_cfg.get("enable", False)):
        print("\n[AUTH] Macro ROC-AUC por identidade (one-vs-all, mesma identidade vs outras):")
        macro_res = auth_macro_auc_per_identity(
            emb=emb,
            P=P,
            y_true=y_eval,
            y_pred=y_pred,
            pmax=pmax,
            rng=rng,
            cfg=macro_auth_cfg,
        )
        if macro_res.get("n_classes_used", 0) > 0:
            print(
                f"  [AUTH] Macro ROC-AUC (média simples em {macro_res['n_classes_used']} classes): "
                f"cosine={macro_res['cos_mean']:.4f} (mediana={macro_res['cos_median']:.4f}) | "
                f"prob_dot={macro_res['prob_mean']:.4f} (mediana={macro_res['prob_median']:.4f}) | "
                f"id+conf={macro_res['conf_mean']:.4f} (mediana={macro_res['conf_median']:.4f})"
            )
        else:
            print("  [AUTH] Macro ROC-AUC: não há classes suficientes (precisa >=2 amostras por classe).")

# imprime resultados (métricas além de acurácia)
    cm_cos = {"TP": int(res_cos.get("TP", 0)), "TN": int(res_cos.get("TN", 0)), "FP": int(res_cos.get("FP", 0)),
              "FN": int(res_cos.get("FN", 0))}
    mets_cos = _binary_metrics_from_counts(cm_cos)
    print("\n[AUTH] Resultados (cosine-threshold):",
          {"thr": float(thr_cos), "mode": res_cos.get("mode"), "pairs": res_cos.get("pairs"), **mets_cos})
    print_confusion_binary("[AUTH] Confusão cosine (same vs different)", cm_cos)

    if "m_f1" in res_prob:
        print("[AUTH] Resultados (prob_dot-threshold):", {"thr": float(thr_prob), **{k: res_prob.get(k) for k in (
        "mode", "pairs", "m_acc", "m_f1", "m_balanced_acc", "TP", "TN", "FP", "FN")}})
        print_confusion_binary("[AUTH] Confusão prob_dot (same vs different)",
                               {"TP": int(res_prob["TP"]), "TN": int(res_prob["TN"]), "FP": int(res_prob["FP"]),
                                "FN": int(res_prob["FN"])})
    else:
        print("[AUTH] Resultados (prob_dot-threshold):", res_prob)

    if "m_f1" in res_conf:
        print("[AUTH] Resultados (id+conf):", {"thr_conf": float(thr_conf), **{k: res_conf.get(k) for k in (
        "mode", "pairs", "m_acc", "m_f1", "m_balanced_acc", "TP", "TN", "FP", "FN")}})
        print_confusion_binary("[AUTH] Confusão id+conf (same vs different)",
                               {"TP": int(res_conf["TP"]), "TN": int(res_conf["TN"]), "FP": int(res_conf["FP"]),
                                "FN": int(res_conf["FN"])})
    else:
        print("[AUTH] Resultados (id+conf):", res_conf)

    # matrizes por âncora (agora: cosine vs id+conf)
    n_anchor = int(auth_cfg.get("n_anchor_images", 10))
    anchor_idx = rng.choice(n, size=min(n_anchor, n), replace=False).astype(np.int64, copy=False)

    print("\n[AUTH] Matrizes por âncora (TP/FP/FN/TN vs todas as outras amostras):")
    for ai in anchor_idx.tolist():
        mask_other = np.ones(n, dtype=bool)
        mask_other[int(ai)] = False
        truth_same = (y_eval == y_eval[int(ai)]) & mask_other

        sims = emb @ emb[int(ai)]
        pred_same_cos = (sims >= float(thr_cos)) & mask_other
        cm_a_cos = _binary_counts_from_pred_truth(pred_same_cos, truth_same)
        mets_a_cos = _binary_metrics_from_counts(cm_a_cos)

        pred_same_conf = ((y_pred == y_pred[int(ai)]) & (
                    np.minimum(pmax, pmax[int(ai)]) >= float(thr_conf))) & mask_other
        cm_a_conf = _binary_counts_from_pred_truth(pred_same_conf, truth_same)
        mets_a_conf = _binary_metrics_from_counts(cm_a_conf)

        print(f"  âncora idx={int(ai)} | true_class={int(y_eval[int(ai)])}")
        print(f"    cosine: thr={float(thr_cos):.4f} | {mets_a_cos} | cm={cm_a_cos}")
        print(f"    id+conf: thr_conf={float(thr_conf):.4f} | {mets_a_conf} | cm={cm_a_conf}")

    # =========================
    # AUC macro (one-vs-rest) - todas as classes
    # =========================
    macro_auc_cfg = SCRIPT_CONFIG.get("macro_auc", {}) or {}
    if bool(macro_auc_cfg.get("enable", True)):
        classes_for_auc = np.unique(y_eval).astype(np.int64, copy=False)
        aucs = []
        use_fast = bool(macro_auc_cfg.get("use_fast_auc", True))
        prog_every = int(macro_auc_cfg.get("progress_every", 50))
        prog_every = max(0, prog_every)
        t0_auc = time.time()
        C = int(classes_for_auc.size)
        for ci, c in enumerate(classes_for_auc.tolist(), start=1):
            if prog_every > 0 and (ci == 1 or (ci % prog_every) == 0 or ci == C):
                elapsed = time.time() - t0_auc
                rate = (ci / elapsed) if elapsed > 0 else 0.0
                eta = ((C - ci) / rate) if rate > 0 else float('nan')
                print(f"[IDENTIFICAÇÃO] Macro AUC OVR: {ci}/{C} | elapsed={elapsed:.1f}s | ETA~{eta:.1f}s")

            idx_k = np.where(classes_modelo == int(c))[0]
            if idx_k.size == 0:
                continue
            k = int(idx_k[0])
            scores = P[:, k]
            truth = (y_eval == int(c)).astype(np.int8)

            if use_fast:
                auc = roc_auc_fast(scores, truth == 1)
                if np.isnan(auc):
                    continue
                aucs.append(float(auc))
            else:
                fpr, tpr, _thrs, auc = roc_curve_simple(scores, truth)
                if np.isnan(auc) or fpr.size == 0:
                    continue
                aucs.append(float(auc))

        if len(aucs) == 0:
            print("\n[IDENTIFICAÇÃO] Macro AUC (one-vs-rest): não foi possível calcular (sem curvas válidas).")
        else:
            macro_auc = float(np.mean(aucs))
            print("\n[IDENTIFICAÇÃO] Macro AUC (one-vs-rest) entre todas as classes (thresholds variados):")
            print(f"  macro_auc={macro_auc:.6f} | n_classes={len(aucs)} | min={min(aucs):.6f} | max={max(aucs):.6f}")

    # INTERATIVO
    if bool(SCRIPT_CONFIG["enable_interactive_queries"]):
        print("\n[INTERATIVO] Ligado. Digite 'q' para sair.")
        while True:
            cmd = input(
                "\nEscolha: (1) Predizer classe por ID | (2) Autenticar por dois IDs | (q) sair : ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "1":
                s = input("Digite o ID da amostra (0..N-1): ").strip()
                if s.lower() in ("q", "quit", "exit"):
                    break
                try:
                    idx = int(s)
                except Exception:
                    print("ID inválido.")
                    continue
                if idx < 0 or idx >= n:
                    print("Fora do intervalo.")
                    continue
                topk = np.argsort(-P[idx])[:5]
                print(f"  y_pred={int(y_pred[idx])} | pmax={float(P[idx].max()):.4f}")
                print("  top5:", [(int(classes_modelo[k]), float(P[idx][k])) for k in topk])

            elif cmd == "2":
                s1 = input("Digite o ID i (0..N-1): ").strip()
                if s1.lower() in ("q", "quit", "exit"):
                    break
                s2 = input("Digite o ID j (0..N-1): ").strip()
                if s2.lower() in ("q", "quit", "exit"):
                    break
                try:
                    i = int(s1);
                    j = int(s2)
                except Exception:
                    print("IDs inválidos.")
                    continue
                if i < 0 or i >= n or j < 0 or j >= n:
                    print("Fora do intervalo.")
                    continue

                sim_cos = float(np.dot(emb[i], emb[j]))
                sim_prob = float(np.dot(P[i], P[j]))
                same_cos = sim_cos >= float(thr_cos)
                same_prob = sim_prob >= float(thr_prob)
                same_conf = (int(y_pred[i]) == int(y_pred[j])) and (float(min(pmax[i], pmax[j])) >= float(thr_conf))

                print(f"  cosine_sim={sim_cos:.4f} | thr_cos={float(thr_cos):.4f} | same? {bool(same_cos)}")
                print(f"  prob_dot={sim_prob:.4f} | thr_prob={float(thr_prob):.4f} | same? {bool(same_prob)}")
                print(
                    f"  id+conf: yhat_i={int(y_pred[i])} (pmax={float(pmax[i]):.4f}) | yhat_j={int(y_pred[j])} (pmax={float(pmax[j]):.4f}) | thr_conf={float(thr_conf):.4f} | same? {bool(same_conf)}")
                if same_conf:
                    print(f"  ==> Prediz MESMA pessoa: classe={int(y_pred[i])}")
                else:
                    print("  ==> Prediz pessoas DIFERENTES")

            else:
                print("Comando inválido (use 1, 2 ou q).")

# ============================================================
# OUTPUT LOGGING (TEE)
# ============================================================
import sys
import io
import traceback
from datetime import datetime
from pathlib import Path

class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

def _run_with_output_log(main_func, log_prefix="mlp_ident_auth_eval"):
    out_dir = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = out_dir / f"{log_prefix}_output_{ts}.txt"

    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[RUN] {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"[SCRIPT] {Path(__file__).name if '__file__' in globals() else 'interactive'}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            sys.stdout = _Tee(old_out, f)
            sys.stderr = _Tee(old_err, f)

            try:
                main_func()
            except SystemExit:
                raise
            except Exception:
                traceback.print_exc()
                raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err

    print(f"[LOG] Output salvo em: {log_path}")


if __name__ == "__main__":
     _run_with_output_log(main, log_prefix="mlp_ident_auth_eval")
