
# -*- coding: utf-8 -*-
"""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Código de avaliação de modelo com tarefas de identificação e autenticação
Consome payload salvo pelo treino (joblib)
Regressão Logística Multinomial (softmax) sobre features HOG

Notas:
- Identificação (multiclasse): prever o ID/classe de cada imagem.
  Métricas típicas: acurácia top-1; AUC ROC macro (one-vs-rest) para referência.
- Autenticação / verificação (binário): decidir se um par de imagens é da mesma identidade.
  O threshold é ajustado em X_tune/y_tune (anti-leakage) e avaliamos em X_eval/y_eval.
"""

#Referências
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
# CONFIGURAÇÕES
# ============================================================

SCRIPT_CONFIG = {
    # nome do arquivo salvo pelo script de treino (na mesma pasta)
    "model_payload_file": "logreg_pm_model_and_classes.joblib",

    # se quiser sobrescrever o caminho do joblib HOG/dataset (senão usa config_snapshot do payload)
    "dataset_path_override": None,  # ex.: r"C:\\...\\celeba_hog_128x128_o9.joblib"

    # aleatoriedade local deste script (não altera o split do treino, só prints/amostragens)
    "seed": 123,

    # identificação
    "one_vs_all_n_classes": 10,

    # ROC one-vs-all para as 10 classes escolhidas acima
    "roc": {
        "enable": True,
        "save_png": True,
        "show_plots": True,
        "png_name": "roc_ova_10classes_logreg.png",
    },

    # AUC macro (one-vs-rest) para todas as classes (ativar/desativar)
    "macro_auc": {
        "enable": True,
        "use_fast_auc": True,
        "progress_every": 200,
    },

    # autenticação
    "auth": {
        "tune_metric": "f1",  # "f1" | "balanced_acc" | "acc"

        # tuning do threshold (amostragem):
        "pos_pairs_per_class": 50,     # tenta gerar até isso por classe (se houver amostras)
        "neg_pairs_total": 20000,      # negativos totais para tuning
        "threshold_grid_q": 401,       # quantis para varrer threshold (>=101 recomendado)

        # avaliação em pares:
        # modo "auto": se N for pequeno faz "full"; caso contrário faz "sample"
        "eval_mode": "auto",              # "auto" | "full" | "sample"
        "full_if_n_leq": 2500,            # se N <= isso, tenta full (O(N^2))
        "sample_pairs_if_large": 300000,  # #pares amostrados se N grande

        # fração dos pares a avaliar (1.0 = usa tudo; <1.0 acelera prototipagem)
        "pairs_fraction": 1.0,

        # Macro ROC-AUC por identidade (one-vs-all) - usa pares por classe/identidade
        "macro_auc": {
            "enable": True,
            "pos_pairs_per_class": 30,
            "neg_pairs_per_class": 60,
            "classes_fraction": 1.0,
            "max_classes": 0,
            "progress_every": 100,
        },

        # matrizes por âncora
        "n_anchor_images": 10,
    },

    # modo interativo (input)
    "enable_interactive_queries": False,
}


# ============================================================
# IO DO DATASET (HOG joblib)
# ============================================================

# Sobre lidar com classes desbalanceadas
# H. He and E. A. Garcia, “Learning from Imbalanced Data,” IEEE TKDE, vol. 21, no. 9, pp. 1263–1284, 2009.

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

        # tenta achar paths/ids (opcional)
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

# Para padronização (z-score)
# C. M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.
# K. P. Murphy, Machine Learning: A Probabilistic Perspective. MIT Press, 2012.

def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xf = X.astype(np.float32, copy=False)
    return ((Xf - mean) / std).astype(np.float32, copy=False)


# ============================================================
# Regressão logística: funções mínimas para inferência
# ============================================================

#Para regressão logística e estabilidade numérica
# C. M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.
# K. P. Murphy, Machine Learning: A Probabilistic Perspective. MIT Press, 2012.

def stable_softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32, copy=False)
    Zm = Z - Z.max(axis=1, keepdims=True)
    np.exp(Zm, out=Zm)
    Zm /= Zm.sum(axis=1, keepdims=True)
    return Zm

# Normalização L2 para similaridade cosseno em reconhecimento facial
# J. Deng et al., “ArcFace” in Proc. IEEE CVPR, 2019.

def _row_norm_forward(A: np.ndarray, eps: float = 1e-8):
    A = A.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(A * A, axis=1, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return A * inv, inv.astype(np.float32, copy=False)


def logreg_forward_proba(X: np.ndarray, modelo: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (P, logits) onde:
      logits = X @ W + b
      P = softmax(logits)
    """
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    W = np.asarray(modelo["W"], dtype=np.float32)
    b = np.asarray(modelo["b"], dtype=np.float32)
    logits = X @ W + b
    P = stable_softmax(logits)
    return P, logits.astype(np.float32, copy=False)


def predict_labels_logreg(X: np.ndarray, modelo: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (y_pred, P, logits). y_pred no espaço das classes originais (IDs).
    """
    classes = np.asarray(modelo["classes"], dtype=np.int64)
    P, logits = logreg_forward_proba(X, modelo)
    idx = np.argmax(P, axis=1).astype(np.int64, copy=False)
    y_pred = classes[idx]
    return y_pred.astype(np.int64, copy=False), P, logits

# Para verificação via embeddings e distância/similaridade
# F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet” in Proc. IEEE CVPR, 2015.
# J. Deng et al., “ArcFace” in Proc. IEEE CVPR, 2019.

def extract_embeddings_logits(X: np.ndarray, modelo: Dict[str, Any]) -> np.ndarray:
    """
    Embedding = logits (X@W+b) L2-normalizado por linha (cosine-ready).
    """
    _P, logits = logreg_forward_proba(X, modelo)
    emb, _ = _row_norm_forward(logits, eps=1e-8)
    return emb.astype(np.float32, copy=False)


# ============================================================
# Métricas / Confusões
# ============================================================

# Referências para métricas
# M. Sokolova and G. Lapalme, “A systematic analysis of performance measures…,” Information Processing & Management, 2009.
# D. M. W. Powers, “Evaluation: From Precision, Recall and F-Measure to ROC” Journal of Machine Learning Technologies, 2011.
# T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006

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

#Referência para autenticação por pares
# G. B. Huang et al., “Labeled Faces in the Wild…,” UMass Amherst, Tech. Rep. 07-49, 2007.

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


def _binary_counts_from_pred_truth(
    pred=None,
    truth=None,
    y_pred=None,
    y_true=None,
    **kwargs
):
    """
    Compatível com chamadas:
      - _binary_counts_from_pred_truth(pred, truth)
      - _binary_counts_from_pred_truth(pred=..., truth=...)
      - _binary_counts_from_pred_truth(y_pred=..., y_true=...)  (como no seu main)
    """
    if pred is None:
        pred = y_pred
    if truth is None:
        truth = y_true
    if pred is None or truth is None:
        raise ValueError("Você precisa fornecer (pred, truth) ou (y_pred, y_true).")

    pred = np.asarray(pred).astype(bool)
    truth = np.asarray(truth).astype(bool)

    TP = int(np.sum(pred & truth))
    TN = int(np.sum((~pred) & (~truth)))
    FP = int(np.sum(pred & (~truth)))
    FN = int(np.sum((~pred) & truth))
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

def _binary_metrics_from_counts(cm: Dict[str, int]) -> Dict[str, float]:
    TP = float(cm["TP"]); TN = float(cm["TN"]); FP = float(cm["FP"]); FN = float(cm["FN"])
    denom = TP + TN + FP + FN
    acc = (TP + TN) / denom if denom > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    tpr = rec
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_acc": float(bal_acc)}


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

# Referência para threshold de confiança binário
# [10] D. M. W. Powers, “Evaluation: From Precision, Recall and F-Measure to ROC…,” Journal of Machine Learning Technologies, 2011.


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

# Implementação das avaliações de autenticação de pares por similaridade/distância com threshold
# F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet” in Proc. IEEE CVPR, 2015.
# T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006.

def eval_auth_pairs_full(emb: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    """
    Avalia todas as combinações (i<j) sem armazenar índices enormes.
    Retorna contagens TP/TN/FP/FN e métricas.
    """
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        sims = emb[i+1:] @ emb[i]  # (n-i-1,)
        pred = sims >= float(thr)
        truth = (y[i+1:] == y[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "TP": TP, "TN": TN, "FP": FP, "FN": FN, **{f"m_{k}": v for k, v in mets.items()}}

# Referência para avaliação em par por scores usando produto escalar
# B. Schölkopf and A. J. Smola, Learning with Kernels. MIT Press, 2002.

def eval_auth_pairs_full_prob(P: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    """Avalia todas as combinações com score = dot(P_i, P_j)."""
    P = np.asarray(P, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        sims = P[i+1:] @ P[i]
        pred = sims >= float(thr)
        truth = (y[i+1:] == y[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "TP": TP, "TN": TN, "FP": FP, "FN": FN, **{f"m_{k}": v for k, v in mets.items()}}


def eval_auth_pairs_full_identity_confidence(y_pred: np.ndarray, pmax: np.ndarray, y_true: np.ndarray,
                                             thr_conf: float) -> Dict[str, Any]:
    """Avalia todas as combinações com regra: same_id_pred & min_conf >= thr_conf."""
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pmax = np.asarray(pmax, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)
    n = int(y_true.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        same_id = (y_pred[i+1:] == y_pred[i])
        conf = np.minimum(pmax[i+1:], pmax[i])
        pred = same_id & (conf >= float(thr_conf))
        truth = (y_true[i+1:] == y_true[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "TP": TP, "TN": TN, "FP": FP, "FN": FN, **{f"m_{k}": v for k, v in mets.items()}}


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


def eval_auth_pairs_sample(emb: np.ndarray, y: np.ndarray, thr: float, rng: np.random.Generator, n_pairs: int):
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    ii, jj = _sample_pairs_uniform(n=n, rng=rng, n_pairs=int(n_pairs))
    truth = (y[ii] == y[jj])
    sims = np.sum(emb[ii] * emb[jj], axis=1).astype(np.float32, copy=False)
    pred = sims >= float(thr)

    cm = _binary_counts_from_pred_truth(pred, truth)
    mets = _binary_metrics_from_counts(cm)
    return {"mode": "sample", "pairs": int(ii.size), "TP": cm["TP"], "TN": cm["TN"], "FP": cm["FP"], "FN": cm["FN"], **{f"m_{k}": v for k, v in mets.items()}}


# ============================================================
# ROC + AUC (sem sklearn)
# ============================================================

def roc_auc_fast(scores, y_true=None, truth=None, y=None, **kwargs):
    """
    ROC-AUC em O(n log n) com tratamento de empates (ties).

    Compatível com chamadas:
      - roc_auc_fast(scores, truth)
      - roc_auc_fast(scores=s, y_true=...)
      - roc_auc_fast(scores=s, truth=...)
      - roc_auc_fast(scores=s, y=...)

    Referência: estatística U de Mann–Whitney / soma de ranks (AUC = P(score_pos > score_neg) + 0.5*P(tie)).
    """
    if y_true is None:
        y_true = truth if truth is not None else y

    if y_true is None:
        raise TypeError("roc_auc_fast: forneça y_true (ou truth/y).")

    scores = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).ravel()

    if y_true.dtype != np.bool_:
        y_true = (y_true.astype(np.int64) == 1)

    if scores.size != y_true.size:
        raise ValueError(f"roc_auc_fast: tamanhos diferentes: scores={scores.size}, y_true={y_true.size}")

    n = int(scores.size)
    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0  # AUC indefinido; aqui devolvemos 0.0 para não quebrar pipeline

    order = np.argsort(scores, kind="mergesort")  # mergesort é estável (bom para ties)
    s_sorted = scores[order]
    y_sorted = y_true[order]

    # Soma de ranks dos positivos com empates: atribui rank médio ao grupo empatado
    rank_sum_pos = 0.0
    rank = 1  # ranks 1..n
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1

        group_size = j - i
        avg_rank = (rank + (rank + group_size - 1)) / 2.0
        pos_in_group = float(np.sum(y_sorted[i:j]))
        rank_sum_pos += pos_in_group * avg_rank

        rank += group_size
        i = j

    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)





def auth_macro_auc_per_identity(
    y=None,
    y_true=None,
    truth=None,
    y_pred=None,
    pred=None,
    pmax=None,
    conf=None,
    emb=None,
    E=None,
    P=None,
    probs=None,
    rng=None,
    random_state=None,
    cfg=None,
    macro_cfg=None,
    **kwargs
) -> Dict[str, Any]:
    """Macro ROC-AUC de AUTENTICAÇÃO por identidade (one-vs-all, em pares).

    Para cada identidade/classe c:
      - Positivos: pares (i,j) com y_true[i]==c e y_true[j]==c
      - Negativos: pares (i,j) com y_true[i]==c e y_true[j]!=c
      - Calcula AUC(c) para 3 scores de verificação:
          1) cosine  : dot de embeddings L2-normalizados (cosine similarity)
          2) prob_dot: dot entre vetores de probabilidades por classe
          3) id+conf : (mesmo ID predito) * min(pmax_i, pmax_j)

    Retorna a MÉDIA (macro) e MEDIANA dos AUCs por identidade para cada score.

    Observação:
      - Isto NÃO é o AUC global (micro-AUC) em pares do dataset inteiro.
      - Macro-AUC dá peso igual para cada identidade, mesmo que algumas tenham mais imagens.
    """
    # ----------------------------
    # Resolver cfg + enable
    # ----------------------------
    if cfg is None:
        cfg = macro_cfg or {}
    cfg = dict(cfg) if isinstance(cfg, dict) else {}

    enable = bool(cfg.get("enable", True))
    if not enable:
        return {"enabled": False}

    # ----------------------------
    # Compatibilidade de nomes (MLP vs LogReg / chamadas antigas)
    # ----------------------------
    if y_true is None:
        y_true = y if y is not None else truth
    if y_pred is None:
        y_pred = pred
    if pmax is None:
        pmax = conf
    if emb is None:
        emb = E
    if P is None:
        P = probs

    if y_true is None or y_pred is None or pmax is None or emb is None or P is None:
        raise ValueError(
            "auth_macro_auc_per_identity: forneça y_true/y, y_pred, pmax (conf), emb (E) e P (probs)."
        )

    # ----------------------------
    # RNG
    # ----------------------------
    if rng is None:
        rng = random_state
    if isinstance(rng, np.random.Generator):
        rg = rng
    elif rng is None:
        rg = np.random.default_rng(123)
    else:
        rg = np.random.default_rng(int(rng))

    # ----------------------------
    # Normalização de entradas
    # ----------------------------
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    pmax = np.asarray(pmax, dtype=np.float32).ravel()
    emb = np.asarray(emb, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)

    n = int(y_true.size)
    if n < 2:
        return {"enabled": True, "n_classes_used": 0}

    if y_pred.size != n or pmax.size != n:
        raise ValueError(f"auth_macro_auc_per_identity: tamanhos incompatíveis: "
                         f"y_true={n}, y_pred={y_pred.size}, pmax={pmax.size}")
    if emb.shape[0] != n or P.shape[0] != n:
        raise ValueError(f"auth_macro_auc_per_identity: tamanhos incompatíveis: "
                         f"emb={emb.shape}, P={P.shape}, esperado N={n} na dim 0")

    # ----------------------------
    # Config
    # ----------------------------
    pos_pairs_per_class = int(cfg.get("pos_pairs_per_class", 30))
    neg_pairs_per_class = int(cfg.get("neg_pairs_per_class", 60))
    classes_fraction = float(cfg.get("classes_fraction", 1.0))
    max_classes = int(cfg.get("max_classes", 0))
    progress_every = int(cfg.get("progress_every", 200))
    return_per_class = bool(cfg.get("return_per_class", False))

    classes = np.unique(y_true)
    if classes.size == 0:
        return {"enabled": True, "n_classes_used": 0}

    # amostra subconjunto de classes, se configurado
    if 0.0 < classes_fraction < 1.0:
        k = max(1, int(round(classes.size * classes_fraction)))
        k = min(k, int(classes.size))
        classes = rg.choice(classes, size=k, replace=False)

    if max_classes > 0 and classes.size > max_classes:
        classes = rg.choice(classes, size=int(max_classes), replace=False)

    per_class_rows = []  # (class_id, auc_cos, auc_prob, auc_conf)
    t0 = time.time()
    total = int(classes.size)

    for t, c in enumerate(classes.tolist(), start=1):
        idx_c = np.flatnonzero(y_true == int(c)).astype(np.int64, copy=False)
        if idx_c.size < 2:
            continue
        idx_o = np.flatnonzero(y_true != int(c)).astype(np.int64, copy=False)
        if idx_o.size < 1:
            continue

        k_pos = int(max(0, pos_pairs_per_class))
        k_neg = int(max(0, neg_pairs_per_class))
        if k_pos <= 0 or k_neg <= 0:
            continue

        # --- amostra positivos (c vs c), com reposição; evita i==j
        ii_pos = idx_c[rg.integers(0, idx_c.size, size=k_pos, dtype=np.int64)]
        jj_pos = idx_c[rg.integers(0, idx_c.size, size=k_pos, dtype=np.int64)]
        mask = (ii_pos == jj_pos)
        while np.any(mask):
            jj_pos[mask] = idx_c[rg.integers(0, idx_c.size, size=int(np.sum(mask)), dtype=np.int64)]
            mask = (ii_pos == jj_pos)

        # --- amostra negativos (c vs ~c)
        ii_neg = idx_c[rg.integers(0, idx_c.size, size=k_neg, dtype=np.int64)]
        jj_neg = idx_o[rg.integers(0, idx_o.size, size=k_neg, dtype=np.int64)]

        ii = np.concatenate([ii_pos, ii_neg], axis=0)
        jj = np.concatenate([jj_pos, jj_neg], axis=0)
        truth_pairs = np.concatenate([
            np.ones((k_pos,), dtype=np.int8),
            np.zeros((k_neg,), dtype=np.int8),
        ], axis=0).astype(bool, copy=False)

        # scores contínuos
        s_cos = np.sum(emb[ii] * emb[jj], axis=1)
        s_prob = np.sum(P[ii] * P[jj], axis=1)
        same_id = (y_pred[ii] == y_pred[jj]).astype(np.float32, copy=False)
        conf_pair = np.minimum(pmax[ii], pmax[jj]).astype(np.float32, copy=False)
        s_conf = (conf_pair * same_id).astype(np.float32, copy=False)

        auc_cos = float(roc_auc_fast(scores=s_cos, y_true=truth_pairs))
        auc_prob = float(roc_auc_fast(scores=s_prob, y_true=truth_pairs))
        auc_conf = float(roc_auc_fast(scores=s_conf, y_true=truth_pairs))
        per_class_rows.append((int(c), auc_cos, auc_prob, auc_conf))

        if progress_every > 0 and (t % progress_every == 0):
            dt = max(1e-6, time.time() - t0)
            rate = t / dt
            remaining = (total - t) / max(1e-9, rate)
            print(
                f"  [AUTH][macro-AUC] classes processadas: {t}/{total} | usadas={len(per_class_rows)} | "
                f"vel={rate:.1f} cls/s | ETA~{remaining:.1f}s"
            )

    if len(per_class_rows) == 0:
        return {"enabled": True, "n_classes_used": 0}

    arr = np.asarray(per_class_rows, dtype=np.float64)
    # colunas: 0=class_id, 1=cos, 2=prob, 3=conf
    cos_vals = arr[:, 1]
    prob_vals = arr[:, 2]
    conf_vals = arr[:, 3]

    out: Dict[str, Any] = {
        "enabled": True,
        "n_classes_used": int(arr.shape[0]),
        "cos_mean": float(np.mean(cos_vals)),
        "prob_mean": float(np.mean(prob_vals)),
        "conf_mean": float(np.mean(conf_vals)),
        "cos_median": float(np.median(cos_vals)),
        "prob_median": float(np.median(prob_vals)),
        "conf_median": float(np.median(conf_vals)),
    }

    if return_per_class:
        out["per_class"] = {
            int(cid): {"cos": float(cos), "prob": float(prob), "conf": float(conf)}
            for (cid, cos, prob, conf) in per_class_rows
        }

    return out



def _pair_dot_scores_batched(
    A,
    ii,
    jj,
    batch=None,
    batch_size=None,
    dtype=np.float32,
    progress_every=100000,
    label="pairs",
):
    """Calcula dot(A[ii], A[jj]) em batches com tracker de progresso.

    Compatível com duas convenções de chamada:
      - batch=...
      - batch_size=... (como no script do MLP)

    Parâmetros:
      A: matriz (N, D) ou (N, K) (embeddings normalizados, probabilidades etc.)
      ii, jj: índices (mesmo tamanho) para formar pares
      batch/batch_size: tamanho do batch; se ambos vierem, batch_size tem prioridade
      dtype: dtype do cálculo/saída
      progress_every: imprime status a cada N pares (0 desativa)
      label: texto no tracker

    Observação:
      Fancy-indexing materializa (batch, D). Ajustamos dinamicamente o batch para evitar
      explosão de memória quando D é grande (ex.: milhares de classes).
    """
    if batch_size is None:
        batch_size = batch
    if batch_size is None:
        batch_size = 50000

    A = np.asarray(A, dtype=dtype)
    ii = np.asarray(ii, dtype=np.int64).ravel()
    jj = np.asarray(jj, dtype=np.int64).ravel()
    n_pairs = int(ii.size)
    out = np.empty((n_pairs,), dtype=dtype)

    if n_pairs == 0:
        return out

    # Ajuste dinâmico de batch (limite ~120MB para Ai/Aj)
    try:
        bytes_per = np.dtype(dtype).itemsize
    except Exception:
        bytes_per = 4
    max_bytes = 120 * 1024 * 1024
    if A.ndim == 2 and A.shape[1] > 0:
        d = int(A.shape[1])
        max_elems = max(1, max_bytes // max(1, bytes_per))
        batch_size = min(int(batch_size), max(1, max_elems // d))
    else:
        batch_size = int(max(1, batch_size))

    progress_every = int(progress_every) if progress_every else 0
    done = 0
    t0 = time.time()

    for start in range(0, n_pairs, batch_size):
        end = min(n_pairs, start + batch_size)
        Ai = A[ii[start:end]]
        Aj = A[jj[start:end]]
        out[start:end] = np.einsum("ij,ij->i", Ai, Aj, optimize=True).astype(dtype, copy=False)

        done = end
        if progress_every and (done == n_pairs or (done % progress_every) == 0):
            pct = 100.0 * done / float(n_pairs)
            dt = max(1e-9, time.time() - t0)
            rate = done / dt
            eta = (n_pairs - done) / max(1e-9, rate)
            print(
                f"    [AUC-pairs] {label}: {done}/{n_pairs} ({pct:.1f}%) | "
                f"{rate:,.0f} pares/s | ETA~{eta:,.1f}s"
            )

    return out





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
    thrs = np.concatenate(([float(np.max(scores) + 1.0)], thrs, [float(np.min(scores) - 1.0)])).astype(np.float32)

    Ppos = float(np.sum(truth == 1))
    Nneg = float(np.sum(truth == 0))
    tpr = []
    fpr = []
    for thr in thrs:
        pred = scores >= float(thr)
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        TP = cm["TP"]; FP = cm["FP"]
        tpr.append(TP / Ppos if Ppos > 0 else 0.0)
        fpr.append(FP / Nneg if Nneg > 0 else 0.0)

    fpr = np.asarray(fpr, dtype=np.float32)
    tpr = np.asarray(tpr, dtype=np.float32)

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = float(np.trapezoid(tpr, fpr)) if fpr.size > 1 else 0.0
    return fpr, tpr, thrs.astype(np.float32), auc


def plot_roc_ova_for_classes(pick_classes, y_true: np.ndarray, proba: np.ndarray,
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
    plt.title("ROC One-vs-All (10 classes) - LogReg")
    plt.legend(loc="lower right")

    if bool(roc_cfg.get("save_png", True)):
        png_name = str(roc_cfg.get("png_name", "roc_ova_10classes_logreg.png"))
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


# ============================================================
# Split / reconstrução de avaliação (idêntico ao MLP Eval)
# ============================================================

# Referências de separação de sets sem leakage
# P. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” in Proc. IJCAI, 1995.
# # S. Kaufman, S. Rosset, and C. Perlich, “Leakage in Data Mining: Formulation, Detection, and Avoidance,” in Proc. ACM KDD, 2012.

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
    classes_modelo_payload = np.asarray(payload["classes_usadas"], dtype=np.int64)
    standardizer = payload.get("standardizer", None)
    conf = payload.get("config_snapshot", {}) or {}

    # usa a ordem de classes do próprio modelo (colunas de W), se houver
    if isinstance(modelo, dict) and "classes" in modelo:
        classes_modelo = np.asarray(modelo["classes"], dtype=np.int64)
        if classes_modelo.size != classes_modelo_payload.size or np.any(classes_modelo != classes_modelo_payload):
            print("[WARN] classes_usadas (payload) difere de modelo['classes']. Usando modelo['classes'] como referência.")
    else:
        classes_modelo = classes_modelo_payload

    print("\n[LOAD] Payload do modelo carregado com sucesso!")
    print("  arquivo:", payload_path)
    print("  n_classes_modelo:", int(classes_modelo.size))

    n_show = min(20, classes_modelo.size)
    print("\n[INFO] 20 IDs de classes (exemplo):")
    print(" ", classes_modelo[:n_show].tolist())

    print("\n[INFO] Exemplos de parâmetros do modelo (LogReg):")
    W = np.asarray(modelo["W"])
    b = np.asarray(modelo["b"])
    print(f"  W: {tuple(W.shape)} | b: {tuple(b.shape)}")
    if "best_hparams" in payload:
        print("  best_hparams(payload):", payload["best_hparams"])
    if "metrics" in payload:
        print("  metrics(payload):", payload["metrics"])
    if "cv_info" in payload:
        print("  cv_info(payload):", payload["cv_info"])
    if "model_kind" in payload:
        print("  model_kind:", payload["model_kind"])

    if standardizer is None or "mean" not in standardizer or "std" not in standardizer:
        raise ValueError("Payload não contém 'standardizer' com mean/std; não dá para padronizar em inferência.")

    dataset_path = SCRIPT_CONFIG.get("dataset_path_override") or conf.get("dataset_path", None)
    if dataset_path is None:
        raise ValueError("Não achei 'dataset_path' no config_snapshot e não há override no SCRIPT_CONFIG.")
    dataset_path = str(dataset_path)

    print("\n[DATA] Carregando dataset HOG joblib...")
    print("  path:", dataset_path)
    X, y, meta = carregar_dataset_joblib(dataset_path)
    paths = meta.get("paths", None)
    ids = meta.get("ids", None)
    print("  X:", X.shape, X.dtype, "| y:", y.shape, y.dtype)
    print("  n_classes_total:", int(np.unique(y).size))

    # reconstruir split de avaliação (teste alinhado)
    print("\n[SPLIT] Reconstruindo conjunto de avaliação (teste alinhado ao modelo)...")
    X_eval, y_eval, _p_eval, _id_eval = build_eval_split_from_payload(
        X=X, y=y,
        paths=paths if isinstance(paths, np.ndarray) else None,
        ids=ids if isinstance(ids, np.ndarray) else None,
        payload_conf=conf,
        standardizer=standardizer,
        classes_modelo=classes_modelo,
    )
    print("  X_eval:", X_eval.shape, X_eval.dtype)
    print("  y_eval:", y_eval.shape, y_eval.dtype, "| classes_present:", int(np.unique(y_eval).size))
    if y_eval.size > 0:
        print(f"  N={int(y_eval.size)} (IDs válidos: 0..{int(y_eval.size)-1})")
    # =========================
    # TUNING (TREINO) para autenticação (anti-leakage)
    # =========================
    print("\n[SPLIT] Reconstruindo conjunto de TUNING (treino alinhado ao modelo) para autenticação (anti-leakage)...")

    idx_global_tune = np.arange(y.size, dtype=np.int64)

    classes_elig_tune = selecionar_classes_elegiveis(y, int(conf["min_amostras_por_classe"]))
    classes_sel_tune = amostrar_classes(classes_elig_tune, float(conf["frac_classes"]), int(conf["seed_classes"]))

    X1_t, y1_t, p1_t, id1_t, _ = filtrar_por_classes(
        X, y, classes_sel_tune,
        paths=paths if isinstance(paths, np.ndarray) else None,
        ids=ids if isinstance(ids, np.ndarray) else None,
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

    X_tune, y_tune, _p_tune, _id_tune, _ = filtrar_por_classes(
        X_train_feat_t, y_train_t, classes_modelo,
        paths=p_train_t, ids=id_train_t, idx_global=None
    )

    print("  X_tune:", X_tune.shape, X_tune.dtype)
    print("  y_tune:", y_tune.shape, y_tune.dtype, "| classes_present:", int(np.unique(y_tune).size))


    # -------------------------
    # Anti-leakage check: nenhum exemplo do tuning pode estar no eval/teste
    # (se o dataset tiver ids/paths).
    # -------------------------
    try:
        if (_id_eval is not None) and (_id_tune is not None):
            inter = np.intersect1d(np.asarray(_id_eval).ravel(), np.asarray(_id_tune).ravel())
            if inter.size > 0:
                raise RuntimeError(f"[LEAKAGE] Detected overlap between TUNING and EVAL sets: {int(inter.size)} samples.")
    except Exception as e:
        # se ids não estiverem disponíveis ou houver problemas de tipo, apenas avisa.
        print(f"[LEAKAGE] overlap check skipped/failed: {e}")


    # =========================
    # IDENTIFICAÇÃO
    # =========================
    y_pred, P, logits = predict_labels_logreg(X_eval, modelo)
    acc = accuracy(y_eval, y_pred)
    print("\n[IDENTIFICAÇÃO] Acurácia no conjunto de avaliação:")
    print(f"  acc={acc:.4f} ({int(np.sum(y_eval == y_pred))}/{int(y_eval.size)})")

    # one-vs-all
    n_ova = int(SCRIPT_CONFIG["one_vs_all_n_classes"])
    classes_present = np.unique(y_eval).astype(np.int64, copy=False)
    n_pick = min(n_ova, int(classes_present.size))
    pick = rng.choice(classes_present, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=np.int64)

    print(f"\n[IDENTIFICAÇÃO] One-vs-all ({n_pick} classes aleatórias): classes={pick.tolist()}")
    for c in pick.tolist():
        cm = confusion_one_vs_all(y_eval, y_pred, int(c))
        print_confusion_binary(title=f"[One-vs-all] classe={int(c)}", cm=cm)

    # ROC (one-vs-all) para as mesmas classes escolhidas acima
    plot_roc_ova_for_classes(
        pick_classes=pick.tolist(),
        y_true=y_eval,
        proba=P,
        classes_modelo=classes_modelo,
        out_dir=out_dir,
        roc_cfg=SCRIPT_CONFIG.get("roc", {}) or {}
    )

    # =========================
    # AUTENTICAÇÃO
    # =========================
    print("\n[AUTH] Extraindo embeddings (logits normalizados)...")
    emb = extract_embeddings_logits(X_eval, modelo)
    pmax = np.max(P, axis=1).astype(np.float32, copy=False)

    auth_cfg = SCRIPT_CONFIG.get("auth", {}) or {}
    tune_metric = str(auth_cfg.get("tune_metric", "f1")).strip().lower()
    if tune_metric not in ("f1", "balanced_acc", "acc"):
        tune_metric = "f1"

    print("[AUTH] Tuning balanceado de thresholds ...")
    # (anti-leakage) o tuning de thresholds deve usar TREINO (X_tune/y_tune), não o TESTE (X_eval/y_eval)
    y_pred_tune, P_tune, logits_tune = predict_labels_logreg(X_tune, modelo)
    emb_tune = extract_embeddings_logits(X_tune, modelo)
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
        sims_cos = _pair_dot_scores_batched(
            A=emb_tune, ii=ii_t, jj=jj_t,
            label="cosine(logits_norm) [tuning]",
            batch_size=100000,
            progress_every=100000,
        ).astype(np.float32, copy=False)
        sims_prob = _pair_dot_scores_batched(
            A=P_tune, ii=ii_t, jj=jj_t,
            label="prob_dot [tuning]",
            batch_size=100000,
            progress_every=100000,
        ).astype(np.float32, copy=False)

        pos_mask = (tt_t == 1)
        neg_mask = (tt_t == 0)
        print(f"  pares tuning: {pair_info} | métrica otimizada: {tune_metric}")

        # print sim_pos vs sim_neg
        print("\n  [SCORES] cosine(logits): pos vs neg")
        print("    pos:", _score_summary(sims_cos[pos_mask]))
        print("    neg:", _score_summary(sims_cos[neg_mask]))
        print("\n  [SCORES] prob_dot(softmax): pos vs neg")
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

        print("\n  [TUNING] cosine(logits):", {k: stats_cos.get(k) for k in ("metric_optimized","best_metric_value","best_thr","best_cm","best_metrics")})
        print("  [TUNING] prob_dot:", {k: stats_prob.get(k) for k in ("metric_optimized","best_metric_value","best_thr","best_cm","best_metrics")})
        print("  [TUNING] id+conf:", {k: stats_conf.get(k) for k in ("metric_optimized","best_metric_value","best_thr","best_cm","best_metrics")})

    eval_mode = str(auth_cfg.get("eval_mode", "auto")).lower()
    full_if_n_leq = int(auth_cfg.get("full_if_n_leq", 2500))
    n = int(y_eval.size)
    use_full = (eval_mode == "full") or (eval_mode == "auto" and n <= full_if_n_leq)
    print(f"\n[AUTH] Avaliação em pares: mode={'full' if use_full else 'sample'} | N={n}")
    # =========================
    # AUTENTICAÇÃO / VERIFICAÇÃO (pares)
    # =========================
    # Pipeline (análogo ao script de MLP):
    # 1) Extraímos "embeddings" (aqui: logits normalizados) e probabilidades por classe.
    # 2) Ajustamos thresholds no conjunto de tuning (X_tune/y_tune), evitando leakage.
    # 3) Avaliamos em pares. Para N grande, amostramos pares para evitar O(N^2).
    # 4) Calculamos AUC ROC em pares com algoritmo O(n log n) (roc_auc_fast), pois
    #    enumerar todos os thresholds únicos não escala quando n_pairs é grande.
    #
    # Referências (fundamentos / contexto):
    # - C. Bishop; Hastie/Tibshirani/Friedman (classificação e probabilidades).
    # - T. Fawcett, 2006 (ROC/AUC).
    # - Schroff et al., "FaceNet" (embeddings + cosine para verificação), 2015.
    # - Deng et al., "ArcFace" / Wang et al., "CosFace" (margens em espaço angular), 2018.
    #
    # Importante: AUC de AUTENTICAÇÃO aqui é em PARES (mesma identidade?),
    # não confundir com AUC macro one-vs-rest de IDENTIFICAÇÃO.
    pairs_fraction = float(auth_cfg.get("pairs_fraction", 1.0))
    if not (0.0 < pairs_fraction <= 1.0):
        pairs_fraction = 1.0

    # número base de pares quando N é grande
    base_pairs = int(auth_cfg.get("sample_pairs_if_large", 300000))
    n_pairs_eval = int(max(1, round(base_pairs * pairs_fraction))) if base_pairs > 0 else 0

    if use_full:
        res_cos = eval_auth_pairs_full(emb, y_eval, thr_cos)
        res_prob = eval_auth_pairs_full_prob(P, y_eval, thr_prob)
        res_conf = eval_auth_pairs_full_identity_confidence(y_pred, pmax, y_eval, thr_conf)

        # ---- AUC em pares amostrados (mesma identidade?) ----
        if n_pairs_eval > 0 and n > 1:
            ii_a, jj_a = _sample_pairs_uniform(n=n, rng=rng, n_pairs=n_pairs_eval)
            truth_a = (y_eval[ii_a] == y_eval[jj_a]).astype(np.int8, copy=False)

            sims_cos_a = _pair_dot_scores_batched(
                A=emb, ii=ii_a, jj=jj_a,
                label="cosine(logits_norm) [AUC]",
                batch_size=100000,
                progress_every=100000,
            ).astype(np.float32, copy=False)

            sims_prob_a = _pair_dot_scores_batched(
                A=P, ii=ii_a, jj=jj_a,
                label="prob_dot [AUC]",
                batch_size=100000,
                progress_every=100000,
            ).astype(np.float32, copy=False)

            same_id_a = (y_pred[ii_a] == y_pred[jj_a])
            conf_a = np.minimum(pmax[ii_a], pmax[jj_a]).astype(np.float32, copy=False)
            score_conf_a = (conf_a * same_id_a.astype(np.float32)).astype(np.float32, copy=False)

            auc_cos = roc_auc_fast(scores=sims_cos_a, y_true=(truth_a == 1))
            auc_prob = roc_auc_fast(scores=sims_prob_a, y_true=(truth_a == 1))
            auc_conf = roc_auc_fast(scores=score_conf_a, y_true=(truth_a == 1))

            print("\n[AUTH] ROC-AUC (pares amostrados):")
            print(f"  cosine(logits_norm): {auc_cos:.6f} | prob_dot: {auc_prob:.6f} | id+conf: {auc_conf:.6f} | n_pairs={int(truth_a.size)}")

        # ---- Macro ROC-AUC por identidade (one-vs-all em pares) ----
        macro_cfg = auth_cfg.get("macro_auc", {}) or {}
        if bool(macro_cfg.get("enable", True)):
            print("\n[AUTH] Macro ROC-AUC por identidade (one-vs-all, em pares):")
            mac = auth_macro_auc_per_identity(
                y=y_eval, y_pred=y_pred, pmax=pmax, emb=emb, P=P, rng=rng, cfg=macro_cfg
            )
            if mac is None or len(mac) == 0 or not bool(mac.get("enabled", True)):
                print("  (não foi possível calcular)")
            else:
                n_ids = int(mac.get("n_classes_used", 0))
                print(
                    f"  cos_mean={float(mac.get('cos_mean', float('nan'))):.6f} | "
                    f"prob_mean={float(mac.get('prob_mean', float('nan'))):.6f} | "
                    f"conf_mean={float(mac.get('conf_mean', float('nan'))):.6f} | "
                    f"n_ids={n_ids}"
                )
                print(
                    f"  cos_median={float(mac.get('cos_median', float('nan'))):.6f} | "
                    f"prob_median={float(mac.get('prob_median', float('nan'))):.6f} | "
                    f"conf_median={float(mac.get('conf_median', float('nan'))):.6f}"
                )
    else:
        # Amostramos UMA vez e reaproveitamos os mesmos pares para:
        # - métricas por threshold (cosine / prob_dot / id+conf)
        # - AUC (threshold-var) para cada score
        if n_pairs_eval <= 0 or n <= 1:
            # fallback (deveria ser raro)
            res_cos = eval_auth_pairs_sample(emb, y_eval, thr_cos, rng=rng, n_pairs=max(1, base_pairs))
            res_prob = {"mode": "sample", "pairs": int(res_cos.get("pairs", 0)), "thr": float(thr_prob), "TP": 0, "TN": 0, "FP": 0, "FN": 0, "m_acc": float("nan"), "m_f1": float("nan"), "m_balanced_acc": float("nan")}
            res_conf = {"mode": "sample", "pairs": int(res_cos.get("pairs", 0)), "thr": float(thr_conf), "TP": 0, "TN": 0, "FP": 0, "FN": 0, "m_acc": float("nan"), "m_f1": float("nan"), "m_balanced_acc": float("nan")}
        else:
            ii_s, jj_s = _sample_pairs_uniform(n=n, rng=rng, n_pairs=n_pairs_eval)
            truth_s = (y_eval[ii_s] == y_eval[jj_s]).astype(np.int8, copy=False)

            # ---- cosine(logits_norm) ----
            sims_cos = _pair_dot_scores_batched(
                A=emb, ii=ii_s, jj=jj_s,
                label="cosine(logits_norm)",
                batch_size=100000,
                progress_every=100000,
            ).astype(np.float32, copy=False)
            pred_cos = sims_cos >= float(thr_cos)
            cm_cos = _binary_counts_from_pred_truth(y_pred=pred_cos, y_true=truth_s)
            mets_cos = _binary_metrics_from_counts(cm_cos)
            res_cos = {"mode": "sample", "pairs": int(ii_s.size), "thr": float(thr_cos), **cm_cos, **{f"m_{k}": float(v) for k, v in mets_cos.items()}}

            # ---- prob_dot ----
            sims_prob = _pair_dot_scores_batched(
                A=P, ii=ii_s, jj=jj_s,
                label="prob_dot",
                batch_size=100000,
                progress_every=100000,
            ).astype(np.float32, copy=False)
            pred_prob = sims_prob >= float(thr_prob)
            cm_prob = _binary_counts_from_pred_truth(y_pred=pred_prob, y_true=truth_s)
            mets_prob = _binary_metrics_from_counts(cm_prob)
            res_prob = {"mode": "sample", "pairs": int(ii_s.size), "thr": float(thr_prob), **cm_prob, **{f"m_{k}": float(v) for k, v in mets_prob.items()}}

            # ---- id+conf ----
            same_id = (y_pred[ii_s] == y_pred[jj_s])
            conf_s = np.minimum(pmax[ii_s], pmax[jj_s]).astype(np.float32, copy=False)
            pred_conf = same_id & (conf_s >= float(thr_conf))
            cm_conf = _binary_counts_from_pred_truth(y_pred=pred_conf, y_true=truth_s)
            mets_conf = _binary_metrics_from_counts(cm_conf)
            res_conf = {"mode": "sample", "pairs": int(ii_s.size), "thr": float(thr_conf), **cm_conf, **{f"m_{k}": float(v) for k, v in mets_conf.items()}}

            # ---- AUCs (pares amostrados) ----
            score_conf = (conf_s * same_id.astype(np.float32)).astype(np.float32, copy=False)
            auc_cos = roc_auc_fast(scores=sims_cos, y_true=(truth_s == 1))
            auc_prob = roc_auc_fast(scores=sims_prob, y_true=(truth_s == 1))
            auc_conf = roc_auc_fast(scores=score_conf, y_true=(truth_s == 1))

            print("\n[AUTH] ROC-AUC (pares amostrados):")
            print(f"  cosine(logits_norm): {auc_cos:.6f} | prob_dot: {auc_prob:.6f} | id+conf: {auc_conf:.6f} | n_pairs={int(truth_s.size)}")

            # ---- Macro ROC-AUC por identidade (one-vs-all em pares) ----
            macro_cfg = auth_cfg.get("macro_auc", {}) or {}
            if bool(macro_cfg.get("enable", True)):
                print("\n[AUTH] Macro ROC-AUC por identidade (one-vs-all, em pares):")
                mac = auth_macro_auc_per_identity(
                    y=y_eval, y_pred=y_pred, pmax=pmax, emb=emb, P=P, rng=rng, cfg=macro_cfg
                )
                if mac is None or len(mac) == 0 or not bool(mac.get("enabled", True)):
                    print("  (não foi possível calcular)")
                else:
                    n_ids = int(mac.get("n_classes_used", 0))
                    print(
                        f"  cos_mean={float(mac.get('cos_mean', float('nan'))):.6f} | "
                        f"prob_mean={float(mac.get('prob_mean', float('nan'))):.6f} | "
                        f"conf_mean={float(mac.get('conf_mean', float('nan'))):.6f} | "
                        f"n_ids={n_ids}"
                    )
                    print(
                        f"  cos_median={float(mac.get('cos_median', float('nan'))):.6f} | "
                        f"prob_median={float(mac.get('prob_median', float('nan'))):.6f} | "
                        f"conf_median={float(mac.get('conf_median', float('nan'))):.6f}"
                    )
    # imprime resultados (métricas além de acurácia)
    print("\n[AUTH] Resultados (cosine(logits)-threshold):", {"thr": float(thr_cos), **{k: res_cos.get(k) for k in ("mode","pairs","m_acc","m_f1","m_balanced_acc","TP","TN","FP","FN")}})
    print_confusion_binary("[AUTH] Confusão cosine(logits) (same vs different)", {"TP": int(res_cos.get("TP", 0)), "TN": int(res_cos.get("TN", 0)), "FP": int(res_cos.get("FP", 0)), "FN": int(res_cos.get("FN", 0))})

    print("[AUTH] Resultados (prob_dot-threshold):", {"thr": float(thr_prob), **{k: res_prob.get(k) for k in ("mode","pairs","m_acc","m_f1","m_balanced_acc","TP","TN","FP","FN")}})
    print_confusion_binary("[AUTH] Confusão prob_dot (same vs different)", {"TP": int(res_prob.get("TP", 0)), "TN": int(res_prob.get("TN", 0)), "FP": int(res_prob.get("FP", 0)), "FN": int(res_prob.get("FN", 0))})

    print("[AUTH] Resultados (id+conf):", {"thr_conf": float(thr_conf), **{k: res_conf.get(k) for k in ("mode","pairs","m_acc","m_f1","m_balanced_acc","TP","TN","FP","FN")}})
    print_confusion_binary("[AUTH] Confusão id+conf (same vs different)", {"TP": int(res_conf.get("TP", 0)), "TN": int(res_conf.get("TN", 0)), "FP": int(res_conf.get("FP", 0)), "FN": int(res_conf.get("FN", 0))})

    # matrizes por âncora (cosine vs id+conf)
    n_anchor = int(auth_cfg.get("n_anchor_images", 10))
    anchor_idx = rng.choice(n, size=min(n_anchor, n), replace=False).astype(np.int64, copy=False) if n > 0 else np.array([], dtype=np.int64)

    print("\n[AUTH] Matrizes por âncora (TP/FP/FN/TN vs todas as outras amostras):")
    for ai in anchor_idx.tolist():
        mask_other = np.ones(n, dtype=bool)
        mask_other[int(ai)] = False
        truth_same = (y_eval == y_eval[int(ai)]) & mask_other

        sims = emb @ emb[int(ai)]
        pred_same_cos = (sims >= float(thr_cos)) & mask_other
        cm_a_cos = _binary_counts_from_pred_truth(pred_same_cos, truth_same)
        mets_a_cos = _binary_metrics_from_counts(cm_a_cos)

        pred_same_conf = ((y_pred == y_pred[int(ai)]) & (np.minimum(pmax, pmax[int(ai)]) >= float(thr_conf))) & mask_other
        cm_a_conf = _binary_counts_from_pred_truth(pred_same_conf, truth_same)
        mets_a_conf = _binary_metrics_from_counts(cm_a_conf)

        print(f"  âncora idx={int(ai)} | true_class={int(y_eval[int(ai)])}")
        print(f"    cosine(logits): thr={float(thr_cos):.4f} | {mets_a_cos} | cm={cm_a_cos}")
        print(f"    id+conf: thr_conf={float(thr_conf):.4f} | {mets_a_conf} | cm={cm_a_conf}")

    # =========================
    # AUC macro (one-vs-rest) - todas as classes
    # =========================
    # Observação: AUC macro aqui é para IDENTIFICAÇÃO (one-vs-rest por classe).
    # Isso NÃO é o mesmo que AUC de AUTENTICAÇÃO (mesma identidade?), que é calculado em pares.
    macro_auc_cfg = SCRIPT_CONFIG.get("macro_auc", {}) or {}
    if bool(macro_auc_cfg.get("enable", True)):
        use_fast_auc = bool(macro_auc_cfg.get("use_fast_auc", True))
        progress_every = int(macro_auc_cfg.get("progress_every", 200))

        classes_for_auc = np.unique(y_eval).astype(np.int64, copy=False)
        aucs: list[float] = []
        t0 = time.time()

        for i_c, c in enumerate(classes_for_auc.tolist(), start=1):
            idx = np.where(classes_modelo == int(c))[0]
            if idx.size == 0:
                continue
            k = int(idx[0])
            scores = P[:, k]
            truth = (y_eval == int(c)).astype(np.int8, copy=False)

            if use_fast_auc:
                auc = roc_auc_fast(scores=scores, y_true=(truth == 1))
            else:
                fpr, tpr, _thrs, auc = roc_curve_simple(scores, truth)
                if fpr.size == 0:
                    continue

            aucs.append(float(auc))

            if progress_every > 0 and (i_c % progress_every == 0):
                dt = time.time() - t0
                print(f"  [AUC][macro] classes_processadas={i_c}/{int(classes_for_auc.size)} | aucs_ok={len(aucs)} | {dt:.1f}s")

        if len(aucs) == 0:
            print("\n[AUC] Não foi possível calcular AUC macro (sem classes/curvas).")
        else:
            macro_auc = float(np.mean(aucs))
            med = float(np.median(aucs))
            print("\n[AUC] Macro AUC (one-vs-rest) - todas as classes:")
            print(f"  macro_auc={macro_auc:.6f} | mediana={med:.6f} | n_classes={len(aucs)} | min={min(aucs):.6f} | max={max(aucs):.6f}")

    # =========================
    # INTERATIVO (opcional)
    # =========================
    if bool(SCRIPT_CONFIG.get("enable_interactive_queries", False)):
        print("\n[INTERATIVO] Ativo. 'q' para sair.")
        while True:
            s = input("\n(1) Digite um ID de amostra (0..N-1) para predizer classe (ou 'q'): ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                idx = int(s)
            except Exception:
                print("  input inválido.")
                continue
            if idx < 0 or idx >= int(y_eval.size):
                print("  ID fora do range.")
                continue
            print(f"  y_true={int(y_eval[idx])} | y_pred={int(y_pred[idx])} | pmax={float(P[idx].max()):.4f}")

            s2 = input("(2) Digite DOIS IDs (i j) para autenticar (ou enter para pular): ").strip()
            if not s2:
                continue
            if s2.lower() in ("q", "quit", "exit"):
                break
            parts = s2.split()
            if len(parts) != 2:
                print("  esperado: i j")
                continue
            try:
                i = int(parts[0]); j = int(parts[1])
            except Exception:
                print("  IDs inválidos.")
                continue
            if i < 0 or j < 0 or i >= int(y_eval.size) or j >= int(y_eval.size):
                print("  IDs fora do range.")
                continue

            # regra id+conf (principal)
            same_pred = (y_pred[i] == y_pred[j]) and (min(float(P[i].max()), float(P[j].max())) >= float(thr_conf))
            if same_pred:
                print(f"  [AUTH id+conf] SAME (pred_class={int(y_pred[i])}) | conf_min={min(float(P[i].max()), float(P[j].max())):.4f} >= {float(thr_conf):.4f}")
            else:
                print(f"  [AUTH id+conf] DIFFERENT | conf_min={min(float(P[i].max()), float(P[j].max())):.4f} | same_id_pred={bool(y_pred[i]==y_pred[j])}")

        print("[INTERATIVO] Encerrado.")

# ============================================================
# OUTPUT LOGGING (TEE) - cole acima do bloco if __name__ == "__main__":
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

def _run_with_output_log(main_func, log_prefix="multilogit_auth_eval"):
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
     _run_with_output_log(main, log_prefix="multilogit_auth_eval")
