# -*- coding: utf-8 -*-
"""
CELEBA HOG -> Softmax Regression "from scratch" (Elastic Net + Armijo)

Refatoração focada em VELOCIDADE:
  1) Removi construção de one-hot por loop Python (era gargalo). Agora usa índices de classe (y_idx).
  2) Treino em mini-batches (SGD) para reduzir pico de memória e evitar softmax gigante (m x K) de uma vez.
  3) Armijo por ÉPOCA em um "probe batch" pequeno (ainda 100% dentro do TREINO/CV), evitando
     recalcular perda em batch gigante dezenas de vezes.
  4) Mantive a normalização por m (tamanho do treino) na loss/grad e no proximal L1.

Observação: Com 2036 classes e d=8100, treinar em 100% das classes com muitas épocas pode ser caro.
Use frac_classes, cv_frac, final_frac, epochs_cv/epochs_final e batch_size_* para prototipar.
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
    # caminho do dataset HOG (joblib com {"X":. "y":.} ou tuple (X,y))
    "dataset_path": r"C:\Users\riosd\PycharmProjects\celeb_identification_prj\data\data\celeba_hog_128x128_o9.joblib",

    # seleção de classes
    "frac_classes": 0.10,            # fração das CLASSES ELEGÍVEIS (ex: 0.005 = 0.5%)
    "seed_classes": 42,
    "min_amostras_por_classe": 25,   # mínimo no dataset inteiro para classe ser elegível

    # split treino/teste
    "test_frac": 0.20,
    "seed_split": 42,
    "min_train_por_classe": 5,       # mínimo no TREINO (após split) p/ manter classe

    # CV
    "k_folds": 5,
    "cv_frac": 0.01,                 # fração do treino p/ CV
    "cv_min_por_classe": None,       # se None, usa k_folds
    "final_frac": 1.00,              # fração do treino p/ treino final
    "final_min_por_classe": 1,       # garante ao menos 1 por classe (evita sumir classe quando final_frac<1)

    # treinamento: separar custo CV vs final
    "epochs_cv": 30,
    "epochs_final": 100,

    # minibatch
    "batch_size_cv": 1024,
    "batch_size_final": 2048,

    # Armijo (por época, em probe batch)
    "armijo_alpha0": 2.0,            # alpha inicial grande
    "armijo_growth": 1.25,           # tenta crescer entre épocas (warm-start)
    "armijo_beta": 0.5,              # backtracking
    "armijo_c1": 1e-4,               # condição de suficiência
    "armijo_max_backtracks": 12,     # bem menor que 50 -> mais rápido
    "armijo_alpha_min": 1e-6,
    "armijo_probe_batch": 2048,      # tamanho do probe batch p/ Armijo (<= batch_size_final costuma ser bom)

    # padronização
    "eps_std": 1e-6,

    # grid de regularização (Elastic Net no W, sem regularizar bias)
    # objective total (m = n_train_total):
    #   loss = CE + (l2/(2m))*||W||^2 + (l1/m)*||W||_1
    "grid_l1": [0.0, 1e-4, 3e-4, 1e-3],
    "grid_l2": [0.0, 1e-4, 3e-4, 1e-3],

    # limitar custo do CV: testar só N combos (l1,l2)
    "max_combos_cv": 8,
    "seed_combos_cv": 42,
    "combo_strategy_cv": "cover",    # "cover" tenta cobrir extremos/miolo

    # avaliação / debug
    "n_exemplos_previsao": 12,
    "top_k_confusao": 10,
    "print_every_batches": 25,       # log durante treino (por época)
    "loss_subsample_max": 2000,      # estimar loss por época em subamostra (mais rápido)
}

# ============================================================
# Utilidades de IO / dataset
# ============================================================

def carregar_dataset(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "X" in obj and "y" in obj:
        X, y = obj["X"], obj["y"]
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, y = obj
    else:
        raise ValueError("Formato do joblib não reconhecido. Esperado dict {'X','y'} ou tuple (X,y).")
    return np.asarray(X), np.asarray(y)

def selecionar_classes_elegiveis(y: np.ndarray, min_amostras: int):
    classes, counts = np.unique(y, return_counts=True)
    mask = counts >= int(min_amostras)
    return classes[mask].astype(np.int64, copy=False)

def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_permitidas: np.ndarray):
    mask = np.isin(y, classes_permitidas)
    return X[mask], y[mask]

def amostrar_classes(classes: np.ndarray, frac: float, seed: int):
    frac = float(frac)
    if frac >= 0.999999:
        return np.array(classes, copy=True)
    rng = np.random.default_rng(int(seed))
    n = len(classes)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    return np.sort(classes[idx]).astype(np.int64, copy=False)

# ============================================================
# Padronização (z-score) - float32
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
# Amostragem estratificada com garantia de mínimo por classe
# ============================================================

def amostrar_com_min_por_classe(
    y: np.ndarray,
    frac: float,
    seed: int,
    min_por_classe: int,
):
    """
    Retorna índices para uma amostra estratificada que GARANTE >= min_por_classe em cada classe.
    Se frac for pequeno demais, a função prioriza o mínimo por classe e pode exceder frac.
    """
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y)
    n = y.shape[0]

    classes, counts = np.unique(y, return_counts=True)
    ok = counts >= int(min_por_classe)
    classes_ok = classes[ok]
    if classes_ok.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx_keep = []
    for c in classes_ok:
        idx_c = np.where(y == c)[0]
        pick = rng.choice(idx_c, size=int(min_por_classe), replace=False)
        idx_keep.append(pick)
    idx_keep = np.concatenate(idx_keep).astype(np.int64, copy=False)

    target = int(round(float(frac) * n))
    if target <= idx_keep.size:
        return np.sort(idx_keep), np.sort(classes_ok).astype(np.int64, copy=False)

    restantes = np.setdiff1d(np.arange(n, dtype=np.int64), idx_keep, assume_unique=False)
    if restantes.size == 0:
        return np.sort(idx_keep), np.sort(classes_ok).astype(np.int64, copy=False)

    y_rest = y[restantes]
    add = min(target - idx_keep.size, restantes.size)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=add, random_state=int(seed))
    (idx_add_rel, _) = next(sss.split(np.zeros(restantes.size), y_rest))
    idx_add = restantes[idx_add_rel]

    idx_final = np.concatenate([idx_keep, idx_add]).astype(np.int64, copy=False)
    return np.sort(idx_final), np.sort(classes_ok).astype(np.int64, copy=False)

# ============================================================
# Softmax / encode de rótulos (y_idx)
# ============================================================

def stable_softmax(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    Zm = Z - Z.max(axis=1, keepdims=True)
    np.exp(Zm, out=Zm)
    Zm /= Zm.sum(axis=1, keepdims=True)
    return Zm

def codificar_rotulos(y: np.ndarray, classes: np.ndarray):
    """
    classes deve estar ordenado e conter todos os rótulos de y.
    Retorna y_idx em [0..K-1].
    """
    classes = np.asarray(classes, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    pos = np.searchsorted(classes, y)
    if np.any(pos < 0) or np.any(pos >= classes.size) or np.any(classes[pos] != y):
        raise ValueError("y contém rótulos fora de classes.")
    return pos.astype(np.int64, copy=False)

# ============================================================
# Objective + grad (data + L2) e proximal L1
# ============================================================

def data_loss_ce(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_idx: np.ndarray):
    """
    Cross-entropy média no batch (sem regularização).
    """
    Z = X @ W + b
    P = stable_softmax(Z)
    eps = np.float32(1e-12)
    ll = -np.log(P[np.arange(P.shape[0]), y_idx] + eps)
    return float(ll.mean())

def reg_loss_elasticnet(W: np.ndarray, l1: float, l2: float, m_total: int):
    m = float(max(int(m_total), 1))
    l1 = float(l1); l2 = float(l2)
    reg = 0.0
    if l2 != 0.0:
        reg += (l2 / (2.0 * m)) * float(np.sum(W * W))
    if l1 != 0.0:
        reg += (l1 / m) * float(np.sum(np.abs(W)))
    return float(reg)

def objective_total(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_idx: np.ndarray, l1: float, l2: float, m_total: int):
    return data_loss_ce(W, b, X, y_idx) + reg_loss_elasticnet(W, l1, l2, m_total)

def batch_grad_ce_l2(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_idx: np.ndarray, l2: float, m_total: int):
    """
    Gradiente do termo suave: CE(batch média) + (l2/(2*m_total))*||W||^2
    OBS: L1 é tratado via proximal.
    """
    B = int(X.shape[0])
    Z = X @ W + b
    P = stable_softmax(Z)
    P[np.arange(B), y_idx] -= 1.0
    P /= np.float32(max(B, 1))  # normalização por batch (SGD)
    dW = X.T @ P
    if float(l2) != 0.0:
        dW += (np.float32(l2) / np.float32(max(m_total, 1))) * W  # normalização por m_total
    db = P.sum(axis=0)
    return dW.astype(np.float32, copy=False), db.astype(np.float32, copy=False)

def proximal_l1(W: np.ndarray, thresh: float):
    thresh = np.float32(thresh)
    return np.sign(W) * np.maximum(np.float32(0.0), np.abs(W) - thresh)

# ============================================================
# Armijo por ÉPOCA (em probe batch) + Treino SGD
# ============================================================

def armijo_alpha_epoch(
    W: np.ndarray, b: np.ndarray,
    X_probe: np.ndarray, y_probe_idx: np.ndarray,
    dW_probe: np.ndarray, db_probe: np.ndarray,
    l1: float, l2: float,
    m_total: int,
    alpha_start: float,
    alpha0: float,
    beta: float,
    c1: float,
    max_backtracks: int,
    alpha_min: float,
):
    """
    Retorna alpha aprovado (>=alpha_min), e quantos backtracks.
    Condição:
      F(W - a*g) <= F(W) - c1 * a * ||g||^2
    F inclui CE no probe + regularização (com normalização por m_total).
    L1 entra via proximal no teste de passo.
    """
    alpha = float(min(alpha_start, alpha0))
    bt = 0

    F_old = objective_total(W, b, X_probe, y_probe_idx, l1=l1, l2=l2, m_total=m_total)
    grad_norm_sq = float(np.sum(dW_probe * dW_probe) + np.sum(db_probe * db_probe))
    if grad_norm_sq <= 0.0:
        return float(max(alpha, alpha_min)), bt, F_old, F_old

    while True:
        W_try = proximal_l1(W - np.float32(alpha) * dW_probe,
                           thresh=(alpha * float(l1)) / float(max(m_total, 1)))
        b_try = b - np.float32(alpha) * db_probe

        F_new = objective_total(W_try, b_try, X_probe, y_probe_idx, l1=l1, l2=l2, m_total=m_total)

        if (F_new <= F_old - float(c1) * alpha * grad_norm_sq) or (alpha <= float(alpha_min)) or (bt >= int(max_backtracks)):
            return float(max(alpha, alpha_min)), bt, F_old, F_new

        alpha *= float(beta)
        bt += 1

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

    alpha_prev = float(CONFIG["armijo_alpha0"])
    hist = []
    alphas = []
    backtracks = []

    batch_size = int(batch_size)
    if batch_size <= 0 or batch_size > m_total:
        batch_size = m_total

    probe_batch = int(CONFIG["armijo_probe_batch"])
    if probe_batch <= 0:
        probe_batch = min(2048, m_total)

    for ep in range(int(epochs)):
        if use_armijo:
            pb = min(probe_batch, m_total)
            probe_idx = rng.choice(m_total, size=pb, replace=False)
            Xp = X_train[probe_idx]
            yp = y_idx[probe_idx]

            dW_p, db_p = batch_grad_ce_l2(W, b, Xp, yp, l2=float(l2), m_total=m_total)

            alpha_start = min(float(CONFIG["armijo_alpha0"]), alpha_prev * float(CONFIG["armijo_growth"]))
            alpha_ep, bt, F_old, F_new = armijo_alpha_epoch(
                W=W, b=b,
                X_probe=Xp, y_probe_idx=yp,
                dW_probe=dW_p, db_probe=db_p,
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
            print(f"[Armijo] Época {ep+1}/{epochs} | alpha={alpha_ep:.3e} | backtracks={bt} | Fprobe {F_old:.4f}->{F_new:.4f}")
        else:
            alpha_ep = float(CONFIG["armijo_alpha0"])
            alphas.append(float(alpha_ep))
            backtracks.append(0)

        a32 = np.float32(alpha_ep)
        shrink = a32 * (np.float32(l1) / np.float32(max(m_total, 1)))

        perm = rng.permutation(m_total)
        n_batches = int(np.ceil(m_total / batch_size))

        for bi, start in enumerate(range(0, m_total, batch_size), start=1):
            batch = perm[start:start + batch_size]
            Xb = X_train[batch]
            yb = y_idx[batch]

            dW, db = batch_grad_ce_l2(W, b, Xb, yb, l2=float(l2), m_total=m_total)
            W -= a32 * dW
            b -= a32 * db
            if float(l1) != 0.0:
                W = proximal_l1(W, thresh=float(shrink))

            if (bi % int(CONFIG["print_every_batches"]) == 0) or (bi == n_batches):
                loss_b = data_loss_ce(W, b, Xb, yb) + reg_loss_elasticnet(W, l1, l2, m_total)
                print(f"[Treino] Época {ep+1}/{epochs} | batch {bi}/{n_batches} | loss~={loss_b:.4f}", end="\r")

        print(" " * 120, end="\r")

        sub_n = min(int(CONFIG["loss_subsample_max"]), m_total)
        sub = rng.choice(m_total, size=sub_n, replace=False)
        loss_ep = objective_total(W, b, X_train[sub], y_idx[sub], l1=l1, l2=l2, m_total=m_total)
        hist.append(float(loss_ep))
        print(f"[SGD] Época {ep+1}/{epochs} | alpha={alpha_ep:.3e} | loss_sub~={loss_ep:.6f}")

    stats = {
        "alpha_epoch": alphas,
        "alpha_mean": float(np.mean(alphas)) if len(alphas) else None,
        "alpha_median": float(np.median(alphas)) if len(alphas) else None,
        "alpha_min": float(np.min(alphas)) if len(alphas) else None,
        "alpha_max": float(np.max(alphas)) if len(alphas) else None,
        "armijo_backtracks_mean": float(np.mean(backtracks)) if len(backtracks) else None,
        "armijo_backtracks_max": int(np.max(backtracks)) if len(backtracks) else None,
        "loss_hist_sub": hist,
    }

    return {"W": W, "b": b, "classes": classes_modelo, "stats": stats}

# ============================================================
# Predição / métricas
# ============================================================

def predict_labels(X: np.ndarray, W: np.ndarray, b: np.ndarray, classes: np.ndarray):
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    Z = X @ W + b
    P = stable_softmax(Z)
    idx = np.argmax(P, axis=1).astype(np.int64, copy=False)
    y_pred = classes[idx]
    return y_pred, P

def report_accuracy(nome: str, y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int64, copy=False)
    y_pred = y_pred.astype(np.int64, copy=False)
    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    print(f"[ACC] {nome}: {acc:.4f} (n={y_true.size})")
    return acc

def mostrar_exemplos_previsao(y_true: np.ndarray, y_pred: np.ndarray, P: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if n <= 0:
        return
    m = y_true.shape[0]
    if m == 0:
        return
    idx = rng.choice(m, size=min(n, m), replace=False)
    print("\n[Exemplos] (verdadeiro -> predito | prob_pred)")
    for i in idx:
        prob = float(np.max(P[i]))
        print(f"  {int(y_true[i])} -> {int(y_pred[i])} | p={prob:.3f}")

def matriz_confusao_top_k(y_true: np.ndarray, y_pred: np.ndarray, top_k: int):
    top_k = int(top_k)
    if top_k <= 0:
        return
    classes, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    top = classes[order[:top_k]]
    mask = np.isin(y_true, top)
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=top)
    print(f"\n[Matriz de Confusão] Top-{top_k} classes (mais frequentes no TESTE filtrado):")
    print("labels:", top.tolist())
    print(cm)

# ============================================================
# CV: escolher melhor (l1,l2) em até 8 combos
# ============================================================

def montar_combos_l1_l2(grid_l1, grid_l2, max_combos: int, strategy: str, seed: int):
    grid_l1 = [float(x) for x in grid_l1]
    grid_l2 = [float(x) for x in grid_l2]
    all_combos = [(l1, l2) for l1 in grid_l1 for l2 in grid_l2]
    if max_combos is None or int(max_combos) <= 0 or len(all_combos) <= int(max_combos):
        return all_combos

    max_combos = int(max_combos)
    rng = np.random.default_rng(int(seed))
    strategy = str(strategy).lower().strip()

    if strategy == "random":
        idx = rng.choice(len(all_combos), size=max_combos, replace=False)
        return [all_combos[i] for i in idx]

    # "cover": tenta pegar extremos e miolo, e completa aleatório
    combos = set()
    combos.add((min(grid_l1), min(grid_l2)))
    combos.add((min(grid_l1), max(grid_l2)))
    combos.add((max(grid_l1), min(grid_l2)))
    combos.add((max(grid_l1), max(grid_l2)))
    combos.add((np.median(grid_l1), np.median(grid_l2)))
    combos.add((np.median(grid_l1), min(grid_l2)))
    combos.add((min(grid_l1), np.median(grid_l2)))
    combos.add((np.median(grid_l1), max(grid_l2)))

    combos = [(float(a), float(b)) for (a, b) in combos]
    valid = set(all_combos)
    combos = [c for c in combos if c in valid]

    if len(combos) >= max_combos:
        return combos[:max_combos]

    rest = [c for c in all_combos if c not in set(combos)]
    need = max_combos - len(combos)
    idx = rng.choice(len(rest), size=need, replace=False)
    combos.extend([rest[i] for i in idx])
    return combos

def escolher_melhores_lambdas_por_cv(
    X_train_feat: np.ndarray,
    y_train_labels: np.ndarray,
    classes_fixas: np.ndarray,
    k_folds: int,
    grid_l1,
    grid_l2,
    epochs: int,
    batch_size: int,
    seed: int,
    max_combos: int,
    combo_strategy: str,
    seed_combos: int,
):
    skf = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=int(seed))

    combos = montar_combos_l1_l2(grid_l1, grid_l2, max_combos=max_combos, strategy=combo_strategy, seed=seed_combos)
    print(f"[CV] combos (l1,l2) testados (max={max_combos}): {combos}")

    best = None
    for (l1, l2) in combos:
        accs = []
        alpha_meds = []
        for fold, (tr, va) in enumerate(skf.split(X_train_feat, y_train_labels), start=1):
            Xtr = X_train_feat[tr]
            ytr = y_train_labels[tr]
            Xva = X_train_feat[va]
            yva = y_train_labels[va]

            modelo = treinar_softmax_elasticnet_sgd(
                X_train=Xtr,
                y_train=ytr,
                classes_modelo=classes_fixas,
                l1=float(l1),
                l2=float(l2),
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=int(seed) + 1000 * fold + 7,
                use_armijo=True,
            )
            yhat, _ = predict_labels(Xva, modelo["W"], modelo["b"], modelo["classes"])
            acc = report_accuracy(f"CV fold {fold} (l1={l1}, l2={l2})", yva, yhat)
            accs.append(acc)

            st = modelo["stats"]
            if st.get("alpha_median") is not None:
                alpha_meds.append(float(st["alpha_median"]))

        mean_acc = float(np.mean(accs)) if len(accs) else -1.0
        alpha_med = float(np.median(alpha_meds)) if len(alpha_meds) else None
        print(f"[CV] (l1={l1}, l2={l2}) mean_acc={mean_acc:.4f} | alpha_med~{alpha_med}")

        if (best is None) or (mean_acc > best[1]):
            best = ((l1, l2), mean_acc, alpha_med)

    return best  # ((l1,l2), mean_acc, alpha_mediana)

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

    # Etapa 1: classes elegíveis e amostragem de classes
    classes_elig = selecionar_classes_elegiveis(y, CONFIG["min_amostras_por_classe"])
    print(f"\n[Etapa 1/7] Classes elegíveis (>= {CONFIG['min_amostras_por_classe']}): {len(classes_elig)}")
    classes_sel = amostrar_classes(classes_elig, CONFIG["frac_classes"], CONFIG["seed_classes"])
    print(f"[Etapa 1/7] Após seleção de classes: {len(classes_sel)} classes (frac={CONFIG['frac_classes']*100:.3f}%)")

    X, y = filtrar_por_classes(X, y, classes_sel)
    print(f"[Etapa 1/7] X={X.shape} | n classes={len(np.unique(y))}")

    # Etapa 2: split treino/teste estratificado
    print(f"\n[Etapa 2/7] Split treino/teste (test_frac={CONFIG['test_frac']:.2f}) ...")
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y,
        test_size=float(CONFIG["test_frac"]),
        random_state=int(CONFIG["seed_split"]),
        stratify=y,
    )
    print(f"[Etapa 2/7] Train(all): X={X_train_all.shape} | classes={len(np.unique(y_train_all))}")
    print(f"[Etapa 2/7] Test (all): X={X_test_all.shape} | classes={len(np.unique(y_test_all))}")

    # Etapa 3: garantir mínimo por classe no TREINO e alinhar TESTE
    min_train = int(max(CONFIG["min_train_por_classe"], CONFIG["k_folds"]))
    print(f"\n[Etapa 3/7] Filtrando classes com >= {min_train} no TREINO ...")
    cls_tr, cts_tr = np.unique(y_train_all, return_counts=True)
    cls_keep = cls_tr[cts_tr >= min_train]
    X_train, y_train = filtrar_por_classes(X_train_all, y_train_all, cls_keep)
    X_test, y_test = filtrar_por_classes(X_test_all, y_test_all, np.unique(y_train))
    print(f"[Etapa 3/7] Train(filtrado): X={X_train.shape} | classes={len(np.unique(y_train))}")
    print(f"[Etapa 3/7] Test (alinhado): X={X_test.shape} | classes={len(np.unique(y_test))}")

    # Etapa 4: padronizar com stats do TREINO
    print("\n[Etapa 4/7] Padronizando features (z-score) com stats do TREINO.")
    mean_tr, std_tr = fit_standardizer(X_train, eps=float(CONFIG["eps_std"]))
    X_train_feat = apply_standardizer(X_train, mean_tr, std_tr)
    X_test_feat = apply_standardizer(X_test, mean_tr, std_tr)
    print(f"[Etapa 4/7] X_train_feat dtype={X_train_feat.dtype} | X_test_feat dtype={X_test_feat.dtype}")

    # Etapa 5: amostra para CV garantindo >=k por classe
    min_cv = int(CONFIG["cv_min_por_classe"] if CONFIG["cv_min_por_classe"] is not None else CONFIG["k_folds"])
    print(f"\n[Etapa 5/7] Amostrando TREINO para CV: frac={CONFIG['cv_frac']:.4f} com min_por_classe={min_cv} ...")
    idx_cv, _ = amostrar_com_min_por_classe(
        y=y_train,
        frac=float(CONFIG["cv_frac"]),
        seed=int(CONFIG["seed_split"]),
        min_por_classe=min_cv,
    )
    if idx_cv.size == 0:
        raise RuntimeError("Amostra para CV ficou vazia. Ajuste cv_frac/min_amostras/k_folds.")
    X_cv = X_train_feat[idx_cv]
    y_cv = y_train[idx_cv]
    _, cts = np.unique(y_cv, return_counts=True)
    print(f"[Etapa 5/7] CV sample: X={X_cv.shape} | classes={len(np.unique(y_cv))}")
    print(f"[Etapa 5/7] min count por classe no CV sample = {int(cts.min())} (deve ser >= {CONFIG['k_folds']})")

    # Etapa 6: grid-search por CV (combos limitados)
    epochs_cv = int(CONFIG["epochs_cv"])
    print(f"\n[Etapa 6/7] Rodando grid-search por CV (Armijo + L1/L2) | epochs_cv={epochs_cv} ...")
    best = escolher_melhores_lambdas_por_cv(
        X_train_feat=X_cv,
        y_train_labels=y_cv,
        classes_fixas=np.unique(y_cv).astype(np.int64, copy=False),
        k_folds=int(CONFIG["k_folds"]),
        grid_l1=CONFIG["grid_l1"],
        grid_l2=CONFIG["grid_l2"],
        epochs=int(epochs_cv),
        batch_size=int(CONFIG["batch_size_cv"]),
        seed=int(CONFIG["seed_split"]),
        max_combos=int(CONFIG["max_combos_cv"]),
        combo_strategy=str(CONFIG["combo_strategy_cv"]),
        seed_combos=int(CONFIG["seed_combos_cv"]),
    )
    (best_l1, best_l2), best_score, best_alpha_med = best
    print(f"\n[CV] Melhor: l1={best_l1} | l2={best_l2} | mean_acc={best_score:.4f} | alpha_mediana~{best_alpha_med}")

    # Etapa 7: treino final (amostra maior) e avaliação no teste
    epochs_final = int(CONFIG["epochs_final"])
    print(f"\n[Etapa 7/7] Treino final em frac={CONFIG['final_frac']:.3f} do treino | epochs_final={epochs_final}.")
    idx_final, _ = amostrar_com_min_por_classe(
        y=y_train,
        frac=float(CONFIG["final_frac"]),
        seed=int(CONFIG["seed_split"]) + 999,
        min_por_classe=int(CONFIG["final_min_por_classe"]),
    )
    X_final = X_train_feat[idx_final]
    y_final = y_train[idx_final]
    classes_final = np.unique(y_final).astype(np.int64, copy=False)

    # garante teste alinhado às classes do modelo final (se final_frac < 1)
    X_test_final, y_test_final = filtrar_por_classes(X_test_feat, y_test, classes_final)

    print(f"[Etapa 7/7] Final sample: X={X_final.shape} | classes={len(classes_final)}")
    print(f"[Etapa 7/7] Teste alinhado ao FINAL: X={X_test_final.shape} | classes={len(np.unique(y_test_final))}")

    modelo_final = treinar_softmax_elasticnet_sgd(
        X_train=X_final,
        y_train=y_final,
        classes_modelo=classes_final,
        l1=float(best_l1),
        l2=float(best_l2),
        epochs=int(epochs_final),
        batch_size=int(CONFIG["batch_size_final"]),
        seed=int(CONFIG["seed_split"]) + 2025,
        use_armijo=True,
    )

    print("\n[Resumo Armijo FINAL]")
    st = modelo_final["stats"]
    print(f"  alpha_mean={st['alpha_mean']:.3e} | alpha_median={st['alpha_median']:.3e} | "
          f"alpha_min={st['alpha_min']:.3e} | alpha_max={st['alpha_max']:.3e}")
    print(f"  backtracks_mean={st['armijo_backtracks_mean']:.2f} | backtracks_max={st['armijo_backtracks_max']}")

    # avaliação
    y_pred_train, _ = predict_labels(X_final, modelo_final["W"], modelo_final["b"], modelo_final["classes"])
    y_pred_test, P_test = predict_labels(X_test_final, modelo_final["W"], modelo_final["b"], modelo_final["classes"])

    report_accuracy("TREINO(final sample)", y_final, y_pred_train)
    report_accuracy("TESTE", y_test_final, y_pred_test)

    mostrar_exemplos_previsao(y_test_final, y_pred_test, P_test, n=CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])
    matriz_confusao_top_k(y_test_final, y_pred_test, top_k=CONFIG["top_k_confusao"])


if __name__ == "__main__":
    main()
