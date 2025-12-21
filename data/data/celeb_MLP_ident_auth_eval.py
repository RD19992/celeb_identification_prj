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
   - escolhe automaticamente um threshold de similaridade por amostragem (max acurácia)
   - avalia acurácia de autenticação em pares (todas as combinações se for pequeno; senão amostra)
   - para 10 âncoras aleatórias, imprime TP/FP/FN/TN vs todas as demais amostras
6) Visualização:
   - tenta carregar e exibir/salvar as 10 imagens âncora (se o joblib tiver paths de imagem acessíveis)
7) Interativo (opcional):
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

    # autenticação
    "auth": {
        # tuning do threshold (amostragem):
        "pos_pairs_per_class": 50,     # tenta gerar até isso por classe (se houver amostras)
        "neg_pairs_total": 20000,      # negativos totais para tuning
        "threshold_grid_q": 401,       # quantis para varrer threshold (>=101 recomendado)

        # avaliação em pares:
        # modo "auto": se N for pequeno faz "full"; caso contrário faz "sample"
        "eval_mode": "auto",           # "auto" | "full" | "sample"
        "full_if_n_leq": 2500,         # se N <= isso, tenta full (O(N^2))
        "sample_pairs_if_large": 300000,  # #pares amostrados se N grande

        # matrizes por âncora
        "n_anchor_images": 10,
    },

    # visualização
    "save_anchor_grid_png": True,
    "anchor_grid_png_name": "auth_anchors_grid.png",

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

    W1 = modelo["W1"]; b1 = modelo["b1"]
    W2 = modelo["W2"]; b2 = modelo["b2"]
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


def tune_threshold_cosine(emb: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                          pos_pairs_per_class: int, neg_pairs_total: int, q_grid: int):
    """
    Escolhe threshold que maximiza acurácia em um conjunto amostrado de pares.
    Retorna (best_thr, stats_dict).
    """
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    pos_pairs = _sample_positive_pairs_per_class(y, rng, per_class=int(pos_pairs_per_class))
    neg_pairs = _sample_negative_pairs(y, rng, total=int(neg_pairs_total))
    pairs = pos_pairs + neg_pairs
    if len(pairs) == 0:
        return 0.5, {"note": "sem pares para tuning; usando thr=0.5", "pairs_used": 0}

    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    tt = np.array([p[2] for p in pairs], dtype=np.int8)

    sims = np.sum(emb[ii] * emb[jj], axis=1).astype(np.float32, copy=False)

    q_grid = int(max(21, q_grid))
    qs = np.linspace(0.0, 1.0, q_grid, dtype=np.float32)
    thrs = np.quantile(sims, qs)
    thrs = np.unique(thrs)

    best_thr = float(thrs[0])
    best_acc = -1.0
    best_cm = None

    for thr in thrs:
        pred = (sims >= thr).astype(np.int8)
        TP = int(np.sum((pred == 1) & (tt == 1)))
        TN = int(np.sum((pred == 0) & (tt == 0)))
        FP = int(np.sum((pred == 1) & (tt == 0)))
        FN = int(np.sum((pred == 0) & (tt == 1)))
        denom = TP + TN + FP + FN
        acc = (TP + TN) / denom if denom > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
            best_cm = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    stats = {
        "pairs_used": int(len(pairs)),
        "pos_pairs": int(len(pos_pairs)),
        "neg_pairs": int(len(neg_pairs)),
        "best_acc_on_tuning_pairs": float(best_acc),
        "best_cm_on_tuning_pairs": best_cm,
        "thr_candidates": int(thrs.size),
        "sim_min": float(np.min(sims)),
        "sim_max": float(np.max(sims)),
        "sim_mean": float(np.mean(sims)),
    }
    return best_thr, stats


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
        sims = emb[i+1:] @ emb[i]  # (n-i-1,)
        pred = sims >= float(thr)
        truth = (y[i+1:] == y[i])
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
# Visualização (opcional)
# ============================================================

def try_show_and_save_anchor_images(paths: Optional[np.ndarray],
                                    anchor_local_indices: np.ndarray,
                                    anchor_labels: np.ndarray,
                                    out_dir: Path,
                                    png_name: str,
                                    save_png: bool):
    if paths is None:
        print("\n[VIS] Dataset joblib não trouxe paths de imagem. Não é possível mostrar imagens originais.")
        return

    paths = np.asarray(paths)
    if paths.ndim != 1:
        print("\n[VIS] Paths têm formato inesperado; pulando visualização.")
        return

    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as e:
        print(f"\n[VIS] Falha ao importar matplotlib/PIL ({e}). Pulando visualização.")
        return

    imgs = []
    for li in anchor_local_indices.tolist():
        p = paths[int(li)]
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except Exception:
            imgs.append(None)

    if all(i is None for i in imgs):
        print("\n[VIS] Nenhuma imagem abriu (paths inválidos/sem acesso). Pulando visualização.")
        return

    n = len(anchor_local_indices)
    cols = 5
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(16, 6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = imgs[i]
        if img is not None:
            ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"id={int(anchor_local_indices[i])}\\ncls={int(anchor_labels[i])}", fontsize=10)
    plt.tight_layout()

    if save_png:
        out_path = out_dir / png_name
        fig.savefig(out_path, dpi=150)
        print(f"\n[VIS] Grid salvo em: {out_path}")

    plt.show()


# ============================================================
# Pipeline principal
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
    W1 = modelo["W1"]; b1 = modelo["b1"]; W2 = modelo["W2"]; b2 = modelo["b2"]
    print(f"  W1: {tuple(W1.shape)} | b1: {tuple(b1.shape)}")
    print(f"  W2: {tuple(W2.shape)} | b2: {tuple(b2.shape)}")
    print(f"  act_hidden: {modelo.get('act_hidden', inference_params.get('act_hidden', '??'))}")
    print(f"  act_output: {modelo.get('act_output', inference_params.get('act_output', '??'))}")
    print(f"  use_layernorm: {inference_params.get('use_layernorm', None)} | layernorm_eps: {inference_params.get('layernorm_eps', None)}")
    print(f"  cosine_softmax_scale: {inference_params.get('cosine_softmax_scale', None)} | cosine_softmax_eps: {inference_params.get('cosine_softmax_eps', None)} | use_bias: {inference_params.get('cosine_softmax_use_bias', None)}")
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
    print("  n_classes_eval:", int(np.unique(y_eval).size))
    if y_eval.size == 0:
        raise RuntimeError("y_eval ficou vazio. Algo não bateu (split/seleção/classes do modelo).")

    print("\n[ID] IMPORTANTE: neste script, o 'ID' de amostra para consultas = índice em X_eval (0..N-1).")
    print(f"     N={int(y_eval.size)} (IDs válidos: 0..{int(y_eval.size)-1})")

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

    # =========================
    # AUTENTICAÇÃO
    # =========================
    print("\n[AUTH] Extraindo embeddings...")
    emb = extract_embeddings(X_eval, modelo, inference_params)

    aconf = SCRIPT_CONFIG["auth"]
    thr, thr_stats = tune_threshold_cosine(
        emb=emb, y=y_eval, rng=rng,
        pos_pairs_per_class=int(aconf["pos_pairs_per_class"]),
        neg_pairs_total=int(aconf["neg_pairs_total"]),
        q_grid=int(aconf["threshold_grid_q"]),
    )
    print("\n[AUTH] Threshold escolhido (por tuning):", float(thr))
    print("[AUTH] Stats tuning:", thr_stats)

    mode = str(aconf["eval_mode"]).lower().strip()
    n = int(y_eval.size)
    if mode == "auto":
        mode = "full" if n <= int(aconf["full_if_n_leq"]) else "sample"

    if mode == "full":
        print("\n[AUTH] Avaliando TODOS os pares (i<j)...")
        auth_stats = eval_auth_pairs_full(emb, y_eval, thr=float(thr))
    else:
        print("\n[AUTH] Avaliando por AMOSTRAGEM de pares (N grande).")
        auth_stats = eval_auth_pairs_sample(
            emb, y_eval, thr=float(thr), rng=rng, n_pairs=int(aconf["sample_pairs_if_large"])
        )

    print("\n[AUTH] Resultado autenticação (pares):")
    print(" ", auth_stats)
    if "TP" in auth_stats:
        print_confusion_binary("[AUTH] Confusão global (same vs different)", {
            "TP": int(auth_stats["TP"]),
            "FP": int(auth_stats["FP"]),
            "FN": int(auth_stats["FN"]),
            "TN": int(auth_stats["TN"]),
        })

    # matrizes por âncora
    n_anchor = int(aconf["n_anchor_images"])
    anchor_idx = rng.choice(n, size=min(n_anchor, n), replace=False).astype(np.int64, copy=False)

    print("\n[AUTH] Matrizes por âncora (TP/FP/FN/TN vs todas as outras amostras):")
    for ai in anchor_idx.tolist():
        sims = emb @ emb[int(ai)]
        mask_other = np.ones(n, dtype=bool)
        mask_other[int(ai)] = False

        pred_same = (sims >= float(thr)) & mask_other
        truth_same = (y_eval == y_eval[int(ai)]) & mask_other

        TP = int(np.sum(pred_same & truth_same))
        TN = int(np.sum((~pred_same) & (~truth_same) & mask_other))
        FP = int(np.sum(pred_same & (~truth_same)))
        FN = int(np.sum((~pred_same) & truth_same))

        print_confusion_binary(
            title=f"[Âncora] id={int(ai)} | cls={int(y_eval[int(ai)])}",
            cm={"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        )

    # VISUALIZAÇÃO
    if int(anchor_idx.size) > 0:
        try_show_and_save_anchor_images(
            paths=paths_eval,
            anchor_local_indices=anchor_idx,
            anchor_labels=y_eval[anchor_idx],
            out_dir=out_dir,
            png_name=str(SCRIPT_CONFIG["anchor_grid_png_name"]),
            save_png=bool(SCRIPT_CONFIG["save_anchor_grid_png"]),
        )

    # INTERATIVO
    if bool(SCRIPT_CONFIG["enable_interactive_queries"]):
        print("\n[INTERATIVO] Ligado. Digite 'q' para sair.")
        while True:
            cmd = input("\nEscolha: (1) Predizer classe por ID | (2) Autenticar por dois IDs | (q) sair : ").strip().lower()
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
                x = X_eval[idx:idx+1]
                yhat, P1 = predict_labels(x, classes_modelo, modelo, inference_params)
                topk = np.argsort(-P1[0])[:5]
                print(f"  y_pred={int(yhat[0])} | pmax={float(P1[0].max()):.4f}")
                print("  top5:")
                for k in topk.tolist():
                    print(f"    cls={int(classes_modelo[k])} p={float(P1[0][k]):.4f}")

            elif cmd == "2":
                s1 = input("Digite o ID da amostra 1: ").strip()
                if s1.lower() in ("q", "quit", "exit"):
                    break
                s2 = input("Digite o ID da amostra 2: ").strip()
                if s2.lower() in ("q", "quit", "exit"):
                    break
                try:
                    i = int(s1); j = int(s2)
                except Exception:
                    print("IDs inválidos.")
                    continue
                if i < 0 or i >= n or j < 0 or j >= n:
                    print("Fora do intervalo.")
                    continue

                sim = float(np.dot(emb[i], emb[j]))
                same = sim >= float(thr)
                yhat_i, Pi = predict_labels(X_eval[i:i+1], classes_modelo, modelo, inference_params)
                yhat_j, Pj = predict_labels(X_eval[j:j+1], classes_modelo, modelo, inference_params)

                print(f"  sim(cos)={sim:.4f} | thr={float(thr):.4f} | same? {bool(same)}")
                print(f"  pred1={int(yhat_i[0])} (pmax={float(Pi[0].max()):.4f}) | pred2={int(yhat_j[0])} (pmax={float(Pj[0].max()):.4f})")

                if same:
                    if int(yhat_i[0]) == int(yhat_j[0]):
                        print(f"  => MESMA classe (pelo modelo): {int(yhat_i[0])}")
                    else:
                        if float(Pi[0].max()) >= float(Pj[0].max()):
                            print(f"  => MESMA pessoa? (pela similaridade) | classe sugerida={int(yhat_i[0])} (mais confiante)")
                        else:
                            print(f"  => MESMA pessoa? (pela similaridade) | classe sugerida={int(yhat_j[0])} (mais confiante)")
                else:
                    print("  => NÃO são da mesma classe (pelo threshold).")

            else:
                print("Comando desconhecido. Use 1, 2 ou q.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Execução falhou:", repr(e))
        sys.exit(1)
