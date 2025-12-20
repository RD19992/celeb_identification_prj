# -*- coding: utf-8 -*-
"""
CELEBA HOG -> Softmax Regression "from scratch" (Elastic Net + Armijo)

VERSÃO OTIMIZADA PARA VELOCIDADE (refatorada):
  1) Remove one-hot com loop Python (gargalo). Usa y_idx via np.searchsorted.
  2) Treino em mini-batches (SGD). Evita softmax gigante de uma vez.
  3) Armijo é feito 1x por ÉPOCA em um "probe batch" pequeno (continua só usando treino/CV).
  4) Mantém normalização /m:
       - grad CE usa média do batch (SGD)
       - L2 e L1 são normalizados por m_total (tamanho do treino usado no modelo).
  5) Corrige edge-case que quebrava quando frac=1.0 (train_size == n_samples no StratifiedShuffleSplit).

DICA DE PERFORMANCE:
  - Se você tiver muitas classes (ex: 2036) e k_folds=5, "min 5 por classe" já força >= 10180 amostras no CV.
    Isso é independente de cv_frac. Para CV mais barato:
      * reduza frac_classes (prototipagem), OU
      * defina CONFIG["cv_max_classes"] (CV em subconjunto de classes).
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "dataset_path": r"C:\Users\riosd\PycharmProjects\celeb_identification_prj\data\data\celeba_hog_128x128_o9.joblib",

    # seleção de classes (prototipagem)
    "frac_classes": 0.10,
    "seed_classes": 42,
    "min_amostras_por_classe": 25,

    # split treino/teste
    "test_frac": 0.20,
    "seed_split": 42,
    "min_train_por_classe": 5,  # pós split

    # CV
    "k_folds": 5,
    "cv_frac": 0.01,
    "cv_min_por_classe": None,    # se None -> usa k_folds
    "cv_max_classes": 600,       # opcional: limita nº de classes usadas no CV (acelera MUITO)

    # treino final
    "final_frac": 1.00,
    "final_min_por_classe": 1,

    # treinamento (separados)
    "epochs_cv": 20,
    "epochs_final": 200,

    # minibatch
    "batch_size_cv": 1024,
    "batch_size_final": 2048,

    # Armijo (por época, usando probe batch)
    "armijo_alpha0": 2.0,
    "armijo_growth": 1.25,
    "armijo_beta": 0.5,
    "armijo_c1": 1e-4,
    "armijo_max_backtracks": 12,
    "armijo_alpha_min": 1e-6,
    "armijo_probe_batch": 2048,

    # padronização
    "eps_std": 1e-6,

    # elastic net grid
    "grid_l1": [0.0, 1e-4, 3e-4, 1e-3],
    "grid_l2": [0.0, 1e-4, 3e-4, 1e-3],
    "max_combos_cv": 8,
    "seed_combos_cv": 42,
    "combo_strategy_cv": "cover",  # cover|random

    # logs / debug
    "print_every_batches": 25,
    "loss_subsample_max": 2000,
    "n_exemplos_previsao": 12,
    "top_k_confusao": 10,
}


# ============================================================
# IO / dataset
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
# Padronização
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
# Amostragem estratificada com mínimo por classe (FIX frac=1.0)
# ============================================================

def amostrar_com_min_por_classe(y: np.ndarray, frac: float, seed: int, min_por_classe: int):
    """
    Retorna (idx_sample, classes_ok). Garante >= min_por_classe por classe (para classes que têm suporte).
    Edge-cases corrigidos:
      - Se frac>=1 -> retorna todos os índices (sem StratifiedShuffleSplit).
      - Se add >= restantes.size -> pega todos os restantes (sem StratifiedShuffleSplit).
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

    # se frac ~ 1, retorna tudo (mas ainda pode filtrar classes com suporte)
    if frac >= 0.999999:
        mask = np.isin(y, classes_ok)
        idx = np.flatnonzero(mask).astype(np.int64, copy=False)
        return idx, np.sort(classes_ok)

    rng = np.random.default_rng(int(seed))

    # pega min_por_classe de cada classe
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

    # FIX: se add cobre tudo, pega tudo sem StratifiedShuffleSplit (evita train_size==n_samples)
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
    """
    Opcional: reduz nº de classes consideradas no CV (acelera), mantendo estratificação.
    Retorna X', y' filtrados.
    """
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
# Softmax / encode de rótulos
# ============================================================

def stable_softmax(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    Zm = Z - Z.max(axis=1, keepdims=True)
    np.exp(Zm, out=Zm)
    Zm /= Zm.sum(axis=1, keepdims=True)
    return Zm


def codificar_rotulos(y: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    pos = np.searchsorted(classes, y)
    if np.any(pos < 0) or np.any(pos >= classes.size) or np.any(classes[pos] != y):
        raise ValueError("y contém rótulos fora de classes.")
    return pos.astype(np.int64, copy=False)


# ============================================================
# Objective / grad (CE + L2) e proximal L1
# ============================================================

def data_loss_ce(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_idx: np.ndarray):
    Z = X @ W + b
    P = stable_softmax(Z)
    eps = np.float32(1e-12)
    ll = -np.log(P[np.arange(P.shape[0]), y_idx] + eps)
    return float(ll.mean())


def reg_loss_elasticnet(W: np.ndarray, l1: float, l2: float, m_total: int):
    m = float(max(int(m_total), 1))
    reg = 0.0
    if float(l2) != 0.0:
        reg += (float(l2) / (2.0 * m)) * float(np.sum(W * W))
    if float(l1) != 0.0:
        reg += (float(l1) / m) * float(np.sum(np.abs(W)))
    return float(reg)


def objective_total(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_idx: np.ndarray, l1: float, l2: float, m_total: int):
    return data_loss_ce(W, b, X, y_idx) + reg_loss_elasticnet(W, l1, l2, m_total)


def batch_grad_ce_l2(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_idx: np.ndarray, l2: float, m_total: int):
    """
    grad do termo suave:
      CE (média do batch) + (l2/(2*m_total))*||W||^2
    """
    B = int(X.shape[0])
    Z = X @ W + b
    P = stable_softmax(Z)
    P[np.arange(B), y_idx] -= 1.0
    P /= np.float32(max(B, 1))  # média do batch
    dW = X.T @ P
    if float(l2) != 0.0:
        dW += (np.float32(l2) / np.float32(max(m_total, 1))) * W  # normaliza por m_total
    db = P.sum(axis=0)
    return dW.astype(np.float32, copy=False), db.astype(np.float32, copy=False)


def proximal_l1(W: np.ndarray, thresh: float):
    thresh = np.float32(thresh)
    return np.sign(W) * np.maximum(np.float32(0.0), np.abs(W) - thresh)


# ============================================================
# Armijo (por época) em probe batch
# ============================================================

def armijo_alpha_epoch(
    W: np.ndarray, b: np.ndarray,
    Xp: np.ndarray, yp_idx: np.ndarray,
    dW_p: np.ndarray, db_p: np.ndarray,
    l1: float, l2: float,
    m_total: int,
    alpha_start: float,
    alpha0: float,
    beta: float,
    c1: float,
    max_backtracks: int,
    alpha_min: float,
):
    alpha = float(min(alpha_start, alpha0))
    F_old = objective_total(W, b, Xp, yp_idx, l1=l1, l2=l2, m_total=m_total)
    g2 = float(np.sum(dW_p * dW_p) + np.sum(db_p * db_p))
    if g2 <= 0.0:
        return max(alpha, alpha_min), 0, F_old, F_old

    bt = 0
    while True:
        W_try = proximal_l1(W - np.float32(alpha) * dW_p, thresh=(alpha * float(l1)) / float(max(m_total, 1)))
        b_try = b - np.float32(alpha) * db_p
        F_new = objective_total(W_try, b_try, Xp, yp_idx, l1=l1, l2=l2, m_total=m_total)

        if (F_new <= F_old - float(c1) * alpha * g2) or (alpha <= float(alpha_min)) or (bt >= int(max_backtracks)):
            return max(alpha, alpha_min), bt, F_old, F_new

        alpha *= float(beta)
        bt += 1


# ============================================================
# Treino SGD + proximal L1 (elastic net)
# ============================================================

def treinar_softmax_elasticnet_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes_modelo: np.ndarray,
    l1: float,
    l2: float,
    epochs: int,
    batch_size: int,
    seed: int,
    use_armijo: bool = True,
):
    rng = np.random.default_rng(int(seed))
    X_train = np.ascontiguousarray(X_train.astype(np.float32, copy=False))
    y_train = y_train.astype(np.int64, copy=False)
    classes_modelo = np.sort(np.unique(classes_modelo)).astype(np.int64, copy=False)

    y_idx = codificar_rotulos(y_train, classes_modelo)

    m_total = int(X_train.shape[0])
    d = int(X_train.shape[1])
    K = int(classes_modelo.shape[0])

    W = (0.01 * rng.standard_normal((d, K))).astype(np.float32)
    b = np.zeros((K,), dtype=np.float32)

    # warm-start alpha por época
    alpha_prev = float(CONFIG["armijo_alpha0"])
    alphas = []
    backtracks = []
    loss_hist = []

    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size > m_total:
        batch_size = m_total

    probe = int(CONFIG["armijo_probe_batch"])
    if probe <= 0:
        probe = min(2048, m_total)

    for ep in range(int(epochs)):
        # decide alpha da época via Armijo em probe batch
        if use_armijo:
            pb = min(probe, m_total)
            probe_idx = rng.choice(m_total, size=pb, replace=False)
            Xp = X_train[probe_idx]
            yp = y_idx[probe_idx]

            dW_p, db_p = batch_grad_ce_l2(W, b, Xp, yp, l2=float(l2), m_total=m_total)
            alpha_start = min(float(CONFIG["armijo_alpha0"]), alpha_prev * float(CONFIG["armijo_growth"]))

            alpha_ep, bt, F_old, F_new = armijo_alpha_epoch(
                W=W, b=b,
                Xp=Xp, yp_idx=yp,
                dW_p=dW_p, db_p=db_p,
                l1=float(l1), l2=float(l2),
                m_total=m_total,
                alpha_start=alpha_start,
                alpha0=float(CONFIG["armijo_alpha0"]),
                beta=float(CONFIG["armijo_beta"]),
                c1=float(CONFIG["armijo_c1"]),
                max_backtracks=int(CONFIG["armijo_max_backtracks"]),
                alpha_min=float(CONFIG["armijo_alpha_min"]),
            )
            alpha_prev = float(alpha_ep)
            alphas.append(float(alpha_ep))
            backtracks.append(int(bt))
            print(f"[Armijo] ep={ep+1:03d}/{epochs} alpha={alpha_ep:.3e} bt={bt} F {F_old:.4f}->{F_new:.4f}")
        else:
            alpha_ep = float(CONFIG["armijo_alpha0"])
            alphas.append(float(alpha_ep))
            backtracks.append(0)

        a32 = np.float32(alpha_ep)
        shrink = a32 * (np.float32(l1) / np.float32(max(m_total, 1)))

        perm = rng.permutation(m_total)
        n_batches = int(np.ceil(m_total / batch_size))

        for bi, start in enumerate(range(0, m_total, batch_size), start=1):
            idx = perm[start:start + batch_size]
            Xb = X_train[idx]
            yb = y_idx[idx]

            dW, db = batch_grad_ce_l2(W, b, Xb, yb, l2=float(l2), m_total=m_total)
            W -= a32 * dW
            b -= a32 * db
            if float(l1) != 0.0:
                W = proximal_l1(W, thresh=float(shrink))

            if bi % int(CONFIG["print_every_batches"]) == 0 or bi == n_batches:
                loss_est = objective_total(W, b, Xb, yb, l1=float(l1), l2=float(l2), m_total=m_total)
                print(f"[Treino] ep={ep+1:03d}/{epochs} batch={bi:04d}/{n_batches} loss~={loss_est:.4f}", end="\r")

        print(" " * 120, end="\r")

        sub_n = min(int(CONFIG["loss_subsample_max"]), m_total)
        sub = rng.choice(m_total, size=sub_n, replace=False)
        loss_ep = objective_total(W, b, X_train[sub], y_idx[sub], l1=float(l1), l2=float(l2), m_total=m_total)
        loss_hist.append(float(loss_ep))
        print(f"[SGD] ep={ep+1:03d}/{epochs} alpha={alpha_ep:.3e} loss_sub~={loss_ep:.6f}")

    stats = {
        "alpha_epoch": alphas,
        "alpha_mean": float(np.mean(alphas)) if alphas else None,
        "alpha_median": float(np.median(alphas)) if alphas else None,
        "alpha_min": float(np.min(alphas)) if alphas else None,
        "alpha_max": float(np.max(alphas)) if alphas else None,
        "armijo_bt_mean": float(np.mean(backtracks)) if backtracks else None,
        "armijo_bt_max": int(np.max(backtracks)) if backtracks else None,
        "loss_hist_sub": loss_hist,
    }
    return {"W": W, "b": b, "classes": classes_modelo, "stats": stats}


# ============================================================
# Predição / métricas
# ============================================================

def predict_labels(X: np.ndarray, W: np.ndarray, b: np.ndarray, classes: np.ndarray):
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    P = stable_softmax(X @ W + b)
    idx = np.argmax(P, axis=1).astype(np.int64, copy=False)
    return classes[idx], P


def report_accuracy(nome: str, y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int64, copy=False)
    y_pred = y_pred.astype(np.int64, copy=False)
    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    print(f"[ACC] {nome}: {acc:.4f} ({int(np.sum(y_true==y_pred))}/{y_true.size})")
    return acc


def mostrar_exemplos_previsao(y_true: np.ndarray, y_pred: np.ndarray, P: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if n <= 0 or y_true.size == 0:
        return
    idx = rng.choice(y_true.size, size=min(n, y_true.size), replace=False)
    print("\n[Exemplos] (y_true -> y_pred | p_max)")
    for i in idx:
        print(f"  {int(y_true[i])} -> {int(y_pred[i])} | p={float(P[i].max()):.3f}")


def matriz_confusao_top_k(y_true: np.ndarray, y_pred: np.ndarray, top_k: int):
    top_k = int(top_k)
    if top_k <= 0 or y_true.size == 0:
        return
    classes, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    top = classes[order[:top_k]]
    mask = np.isin(y_true, top)
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=top)
    print(f"\n[Matriz de Confusão] Top-{top_k} classes (TESTE filtrado):")
    print("labels:", top.tolist())
    print(cm)


# ============================================================
# CV: combos L1/L2 limitados (8 combos)
# ============================================================

def montar_combos_l1_l2(grid_l1, grid_l2, max_combos: int, strategy: str, seed: int):
    grid_l1 = [float(x) for x in grid_l1]
    grid_l2 = [float(x) for x in grid_l2]
    full = [(a, b) for a in grid_l1 for b in grid_l2]

    if max_combos is None or int(max_combos) <= 0 or len(full) <= int(max_combos):
        return full

    max_combos = int(max_combos)
    rng = np.random.default_rng(int(seed))
    strategy = str(strategy).lower().strip()

    if strategy == "random":
        idx = rng.choice(len(full), size=max_combos, replace=False)
        return [full[i] for i in idx]

    # cover
    combos = []
    def add(p):
        if p in full and p not in combos:
            combos.append(p)

    g1 = sorted(set(grid_l1))
    g2 = sorted(set(grid_l2))
    i0, im, i1 = 0, len(g1)//2, len(g1)-1
    j0, jm, j1 = 0, len(g2)//2, len(g2)-1

    base = [
        (g1[i0], g2[j0]), (g1[i0], g2[jm]), (g1[i0], g2[j1]),
        (g1[im], g2[j0]), (g1[im], g2[jm]), (g1[im], g2[j1]),
        (g1[i1], g2[j0]), (g1[i1], g2[j1]),
    ]
    for p in base:
        add(p)

    if len(combos) >= max_combos:
        return combos[:max_combos]

    rest = [p for p in full if p not in combos]
    rng.shuffle(rest)
    combos.extend(rest[:max_combos - len(combos)])
    return combos[:max_combos]


def escolher_melhores_lambdas_por_cv(
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    k_folds: int,
    grid_l1,
    grid_l2,
    epochs_cv: int,
    batch_size_cv: int,
    seed: int,
):
    skf = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=int(seed))
    classes_cv = np.unique(y_cv).astype(np.int64, copy=False)

    combos = montar_combos_l1_l2(
        grid_l1=grid_l1,
        grid_l2=grid_l2,
        max_combos=int(CONFIG["max_combos_cv"]),
        strategy=str(CONFIG["combo_strategy_cv"]),
        seed=int(CONFIG["seed_combos_cv"]),
    )
    print(f"[CV] combos testados ({len(combos)}): {combos}")

    best = None  # ((l1,l2), acc, alpha_med)

    for (l1, l2) in combos:
        accs = []
        alpha_meds = []
        for fold, (tr, va) in enumerate(skf.split(X_cv, y_cv), start=1):
            Xtr, ytr = X_cv[tr], y_cv[tr]
            Xva, yva = X_cv[va], y_cv[va]

            modelo = treinar_softmax_elasticnet_sgd(
                X_train=Xtr,
                y_train=ytr,
                classes_modelo=classes_cv,
                l1=float(l1), l2=float(l2),
                epochs=int(epochs_cv),
                batch_size=int(batch_size_cv),
                seed=int(seed) + 1000*fold + 7,
                use_armijo=True,
            )
            yhat, _ = predict_labels(Xva, modelo["W"], modelo["b"], modelo["classes"])
            acc = float(np.mean(yhat == yva))
            accs.append(acc)

            am = modelo["stats"].get("alpha_median")
            if am is not None:
                alpha_meds.append(float(am))

            print(f"  [CV] l1={l1} l2={l2} fold={fold}/{k_folds} acc={acc:.4f}")

        mean_acc = float(np.mean(accs)) if accs else -1.0
        alpha_med = float(np.median(alpha_meds)) if alpha_meds else None
        print(f"[CV] l1={l1} l2={l2} -> mean_acc={mean_acc:.4f} | alpha_med~{alpha_med}")

        if best is None or mean_acc > best[1]:
            best = ((float(l1), float(l2)), mean_acc, alpha_med)

    return best


# ============================================================
# MAIN
# ============================================================

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
    min_cv = int(CONFIG["cv_min_por_classe"] if CONFIG["cv_min_por_classe"] is not None else k)

    # opcional: limita classes para CV (acelera)
    X_train_cv_src, y_train_cv_src = limitar_classes_para_cv(X_train_feat, y_train, CONFIG["cv_max_classes"], seed=int(CONFIG["seed_split"]))

    idx_cv, _ = amostrar_com_min_por_classe(
        y=y_train_cv_src,
        frac=float(CONFIG["cv_frac"]),
        seed=int(CONFIG["seed_split"]),
        min_por_classe=min_cv,
    )
    if idx_cv.size == 0:
        raise RuntimeError("Amostra CV vazia. Ajuste cv_frac/cv_max_classes/k_folds/min_amostras.")
    X_cv = X_train_cv_src[idx_cv]
    y_cv = y_train_cv_src[idx_cv]
    _, cts_cv = np.unique(y_cv, return_counts=True)
    print(f"\n[Etapa 5] CV sample: X={X_cv.shape} | classes={len(np.unique(y_cv))} | min_count={int(cts_cv.min())}")
    target = int(np.ceil(float(CONFIG["cv_frac"]) * y_train_cv_src.shape[0]))
    min_needed = int(len(np.unique(y_train_cv_src)) * min_cv)
    if target < min_needed:
        print(f"[Aviso] cv_frac implica alvo~{target}, mas mínimo por classe exige >= {min_needed}. "
              f"O CV vai usar pelo menos {min_needed} amostras. Para acelerar: reduza frac_classes ou use cv_max_classes.")

    # 6) CV grid-search (8 combos)
    print(f"\n[Etapa 6] CV grid-search | epochs_cv={CONFIG['epochs_cv']} batch_size_cv={CONFIG['batch_size_cv']}")
    (best_l1, best_l2), best_acc, alpha_med = escolher_melhores_lambdas_por_cv(
        X_cv=X_cv,
        y_cv=y_cv,
        k_folds=k,
        grid_l1=CONFIG["grid_l1"],
        grid_l2=CONFIG["grid_l2"],
        epochs_cv=int(CONFIG["epochs_cv"]),
        batch_size_cv=int(CONFIG["batch_size_cv"]),
        seed=int(CONFIG["seed_split"]),
    )
    print(f"\n[CV] Melhor: l1={best_l1} l2={best_l2} mean_acc={best_acc:.4f} alpha_med~{alpha_med}")

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

    # alinhar teste às classes do modelo final (se final_frac<1)
    X_test_final, y_test_final = filtrar_por_classes(X_test_feat, y_test, classes_final)

    print(f"[Etapa 7] Final sample: X={X_final.shape} | classes={len(classes_final)}")
    print(f"[Etapa 7] Test alinhado: X={X_test_final.shape} | classes={len(np.unique(y_test_final))}")

    modelo = treinar_softmax_elasticnet_sgd(
        X_train=X_final,
        y_train=y_final,
        classes_modelo=classes_final,
        l1=float(best_l1),
        l2=float(best_l2),
        epochs=int(CONFIG["epochs_final"]),
        batch_size=int(CONFIG["batch_size_final"]),
        seed=int(CONFIG["seed_split"]) + 2025,
        use_armijo=True,
    )

    st = modelo["stats"]
    print("\n[Armijo FINAL] alpha_median=", st["alpha_median"], " bt_mean=", st["armijo_bt_mean"], " bt_max=", st["armijo_bt_max"])

    yhat_tr, _ = predict_labels(X_final, modelo["W"], modelo["b"], modelo["classes"])
    yhat_te, P_te = predict_labels(X_test_final, modelo["W"], modelo["b"], modelo["classes"])

    report_accuracy("TREINO(final sample)", y_final, yhat_tr)
    report_accuracy("TESTE", y_test_final, yhat_te)
    mostrar_exemplos_previsao(y_test_final, yhat_te, P_te, n=int(CONFIG["n_exemplos_previsao"]), seed=int(CONFIG["seed_split"]))
    matriz_confusao_top_k(y_test_final, yhat_te, top_k=int(CONFIG["top_k_confusao"]))


if __name__ == "__main__":
    main()
