# -*- coding: utf-8 -*-
"""
MLP PM - Avaliação (Identificação + Autenticação) - SCRIPT NOVO
==============================================================

O que este script faz (em ordem):
1) Lê o arquivo salvo pelo treino: "mlp_pm_model_and_classes.joblib" (na MESMA pasta).
2) Confirma leitura e imprime:
   - 20 IDs de classes presentes no modelo
   - exemplos de parâmetros do modelo (shapes, ativações, cosine-softmax, etc.)
3) Reconstroi o conjunto de avaliação (reproduz passos do treino para obter o TESTE alinhado):
   - carrega o joblib HOG usado no treino (path vem do config_snapshot salvo no payload)
   - refaz seleção de classes + split treino/teste (mesmas seeds)
   - aplica o standardizer salvo (mean/std do treino)
   - filtra o teste para as classes usadas no modelo
4) Identificação:
   - prediz classe para cada amostra do conjunto de avaliação
   - calcula acurácia
   - imprime 10 matrizes de confusão one-vs-all para 10 classes aleatórias
5) Autenticação (mesma pessoa?):
   - extrai embeddings do MLP (vetor da camada escondida, normalizado)
   - tuning balanceado de thresholds com F1 / balanced-acc (cosine e prob_dot)
   - imprime sim_pos vs sim_neg para inspeção rápida
   - regra principal: verificação por identidade prevista + confiança (min(pmax_i,pmax_j) >= thr_conf)
   - avalia em pares (full se N pequeno; senão amostra) e imprime métricas + confusão
6) Interativo (opcional):
   - consulta por "ID" de amostra para predizer classe
   - consulta por DOIS IDs para dizer se são da mesma classe (e qual) ou não

Notas IMPORTANTES:
- "ID" aqui significa o índice da amostra dentro do conjunto de avaliação construído por este script
  (0..N-1). O script imprime como mapear.
- Se o seu joblib HOG não contém paths de imagem, a etapa de visualização não consegue mostrar
  as imagens originais (HOG não guarda pixels). Neste caso, o script salva/mostra apenas índices.

Dependências: numpy, joblib, scikit-learn (apenas train_test_split), matplotlib (opcional para visual).
"""

from __future__ import annotations

import sys
import math
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG DO SCRIPT (ajuste aqui)
# ============================================================

SCRIPT_CONFIG = {
    # nome do arquivo salvo pelo script de treino (na mesma pasta)
    "model_payload_file": "mlp_pm_model_and_classes.joblib",

    # se quiser sobrescrever o caminho do joblib HOG/dataset (senão usa config_snapshot do payload)
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

    # AUC macro (one-vs-rest) para todas as classes (ativar/desativar)
    "macro_auc": {
        "enable": True,
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

        # matrizes por âncora
        "n_anchor_images": 10,
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
# MLP: Funções mínimas para inferência (COPIADAS do treino)
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
    auc = float(np.trapz(tpr, fpr)) if fpr.size > 1 else 0.0
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
    # IDENTIFICAÇÃO
    # =========================
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
    # AUTENTICAÇÃO
    # =========================
    print("\n[AUTH] Extraindo embeddings...")
    emb = extract_embeddings(X_eval, modelo, inference_params)
    pmax = np.max(P, axis=1).astype(np.float32, copy=False)

    auth_cfg = SCRIPT_CONFIG.get("auth", {}) or {}
    tune_metric = str(auth_cfg.get("tune_metric", "f1")).strip().lower()
    if tune_metric not in ("f1", "balanced_acc", "acc"):
        tune_metric = "f1"

    print("[AUTH] Tuning balanceado de thresholds ...")
    ii_t, jj_t, tt_t, pair_info = build_tuning_pairs(
        y_eval, rng=rng,
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
        sims_cos = np.sum(emb[ii_t] * emb[jj_t], axis=1).astype(np.float32, copy=False)
        sims_prob = np.sum(P[ii_t] * P[jj_t], axis=1).astype(np.float32, copy=False)

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
            y_pred=y_pred, pmax=pmax,
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
    else:
        res_cos = eval_auth_pairs_sample(emb, y_eval, thr=float(thr_cos), rng=rng,
                                         n_pairs=int(auth_cfg.get("sample_pairs_if_large", 300000)))

        n_pairs = int(auth_cfg.get("sample_pairs_if_large", 300000))
        ii_s, jj_s = _sample_pairs_uniform(n=int(n), rng=rng, n_pairs=int(n_pairs))
        truth_s = (y_eval[ii_s] == y_eval[jj_s]).astype(np.int8)

        sims_prob = np.sum(P[ii_s] * P[jj_s], axis=1).astype(np.float32, copy=False)
        pred_prob = sims_prob >= float(thr_prob)
        cm_prob = _binary_counts_from_pred_truth(pred_prob, truth_s == 1)
        mets_prob = _binary_metrics_from_counts(cm_prob)
        res_prob = {"mode": "sample", "pairs": int(ii_s.size), "TP": cm_prob["TP"], "TN": cm_prob["TN"],
                    "FP": cm_prob["FP"], "FN": cm_prob["FN"], **{f"m_{k}": v for k, v in mets_prob.items()}}

        same_id = (y_pred[ii_s] == y_pred[jj_s])
        conf_s = np.minimum(pmax[ii_s], pmax[jj_s])
        pred_conf = same_id & (conf_s >= float(thr_conf))
        cm_conf = _binary_counts_from_pred_truth(pred_conf, truth_s == 1)
        mets_conf = _binary_metrics_from_counts(cm_conf)
        res_conf = {"mode": "sample", "pairs": int(ii_s.size), "TP": cm_conf["TP"], "TN": cm_conf["TN"],
                    "FP": cm_conf["FP"], "FN": cm_conf["FN"], **{f"m_{k}": v for k, v in mets_conf.items()}}

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
        for c in classes_for_auc.tolist():
            idx_k = np.where(classes_modelo == int(c))[0]
            if idx_k.size == 0:
                continue
            k = int(idx_k[0])
            scores = P[:, k]
            truth = (y_eval == int(c)).astype(np.int8)
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


if __name__ == "__main__":
    main()
