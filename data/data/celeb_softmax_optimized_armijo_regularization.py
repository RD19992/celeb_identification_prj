# -*- coding: utf-8 -*-
"""
CELEBA HOG -> Softmax Regression "from scratch" (com Armijo)
- CV em amostra estratificada do treino (cv_frac)
- Treino final em amostra maior do treino (final_frac)
- Evita warning do StratifiedKFold (classe com 4 membros quando k=5)
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # caminho do dataset HOG (joblib com {"X":..., "y":...} ou tuple (X,y))
    "dataset_path": r"C:\Users\riosd\PycharmProjects\celeb_identification_prj\data\data\celeba_hog_128x128_o9.joblib",

    # prototipagem / seleção de classes
    "frac_classes": 1.00,            # fração das CLASSES ELEGÍVEIS (ex: 0.005 = 0.5%)
    "seed_classes": 42,
    "min_amostras_por_classe": 25,   # mínimo no dataset inteiro para classe ser elegível

    # split treino/teste
    "test_frac": 0.20,
    "seed_split": 42,
    "min_train_por_classe": 5,       # mínimo no TREINO (após split) p/ manter classe

    # CV
    "k_folds": 5,                    # ajuste aqui (ex: 3, 5)
    "cv_frac": 0.01,                 # 1% do treino p/ CV
    "final_frac": 1.00,              # 100% do treino p/ treino final

    # treinamento
    # Separar nº de épocas entre CV e treino final para controlar custo
    "epochs_cv": 30,                 # <= use menor no CV para não explodir custo
    "epochs_final": 100,              # <= use maior (ou igual) no treino final
    "alpha_init": 2.0,               # alpha inicial maior (Armijo vai reduzir se precisar)
    "armijo_beta": 0.5,              # fator de backtracking
    "armijo_sigma": 1e-4,            # parâmetro de suficiência (Armijo)
    "eps_std": 1e-6,

    # grid de regularização (Elastic Net no W, sem regularizar bias)
    # loss = CE + (l2/(2m))*||W||^2 + (l1/m)*||W||_1
    "grid_l1": [0.0, 1e-4, 3e-4, 1e-3],
    "grid_l2": [0.0, 1e-4, 3e-4, 1e-3],

    # limitar custo do CV: testar só N combos (l1,l2)
    "max_combos_cv": 8,              # limita nº de combinações (l1,l2) no CV
    "seed_combos_cv": 42,            # seed p/ seleção das combinações
    "combo_strategy_cv": "cover",    # "cover" cobre extremos/miolo; "random" amostra aleatório

    # outputs
    "n_exemplos_previsao": 10,
    "top_k_confusao": 10,
}


# ============================================================
# Utils: carregamento / checks
# ============================================================

def carregar_joblib_dataset(path_joblib: str):
    obj = joblib.load(path_joblib)
    if isinstance(obj, dict) and ("X" in obj) and ("y" in obj):
        X, y = obj["X"], obj["y"]
    elif isinstance(obj, (list, tuple)) and len(obj) == 2:
        X, y = obj
    else:
        raise ValueError("Formato inesperado do joblib. Esperado dict{'X','y'} ou tuple(X,y).")
    return X, y


def diagnostico_hog_dim(d: int):
    print(f"\n[Diagnóstico HOG] Dimensão de features (d) = {d}")
    if d == 8100:
        print("[Diagnóstico HOG] d=8100 sugere HOG 128x128 (o9, cell8, block2).")
    elif d == 2025:
        print("[Diagnóstico HOG] d=2025 sugere HOG 64x64 (o9, cell8, block2).")
    else:
        print("[Diagnóstico HOG] d não reconhecido automaticamente (ok).")


# ============================================================
# Split / amostragens
# ============================================================

def selecionar_classes_eligiveis(y: np.ndarray, min_amostras: int):
    classes, counts = np.unique(y, return_counts=True)
    mask = counts >= min_amostras
    return classes[mask]


def escolher_classes(y: np.ndarray, frac_classes: float, seed: int, min_amostras: int):
    classes_eligiveis = selecionar_classes_eligiveis(y, min_amostras=min_amostras)
    rng = np.random.default_rng(seed)
    n = len(classes_eligiveis)
    k = max(1, int(np.ceil(frac_classes * n)))
    chosen = rng.choice(classes_eligiveis, size=k, replace=False)
    return np.sort(chosen), classes_eligiveis


def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_escolhidas: np.ndarray):
    mask = np.isin(y, classes_escolhidas)
    return X[mask], y[mask]


def filtrar_min_train_por_classe(X_train: np.ndarray, y_train: np.ndarray, min_train: int):
    classes, counts = np.unique(y_train, return_counts=True)
    keep = classes[counts >= min_train]
    mask = np.isin(y_train, keep)
    return X_train[mask], y_train[mask], keep


def amostrar_estratificado(y: np.ndarray, frac: float, seed: int):
    if frac >= 1.0:
        return np.arange(len(y), dtype=np.int64)
    if frac <= 0.0:
        return np.array([], dtype=np.int64)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=None, train_size=frac, random_state=seed)
    idx, _ = next(sss.split(np.zeros_like(y), y))
    return idx.astype(np.int64)


# ============================================================
# Softmax: forward / loss / grad (elastic net)
# ============================================================

def stable_softmax(Z: np.ndarray):
    Zm = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Zm)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def one_hot(y: np.ndarray, classes: np.ndarray):
    # classes fixas -> colunas fixas
    class_to_col = {c: i for i, c in enumerate(classes.tolist())}
    m = y.shape[0]
    k = classes.shape[0]
    Y = np.zeros((m, k), dtype=np.float32)
    for i in range(m):
        Y[i, class_to_col[int(y[i])]] = 1.0
    return Y


def softmax_loss(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray, classes: np.ndarray, l1: float, l2: float):
    m = X.shape[0]
    Z = X @ W + b
    P = stable_softmax(Z)
    Y = one_hot(y, classes)
    eps = 1e-12
    ce = -np.sum(Y * np.log(P + eps)) / m

    # L2: (l2/(2m))*||W||^2
    # L1: (l1/m)*||W||_1
    reg_l2 = (l2 / (2.0 * m)) * float(np.sum(W * W))
    reg_l1 = (l1 / m) * float(np.sum(np.abs(W)))
    return ce + reg_l2 + reg_l1, P


def softmax_grad(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray, classes: np.ndarray, l2: float):
    # grad do CE + L2 (L1 tratado via proximal)
    m = X.shape[0]
    Z = X @ W + b
    P = stable_softmax(Z)
    Y = one_hot(y, classes)
    dZ = (P - Y) / m
    dW = X.T @ dZ + (l2 / m) * W
    db = np.sum(dZ, axis=0)
    return dW.astype(np.float32), db.astype(np.float32), P


def prox_l1(W: np.ndarray, thresh: float):
    # soft-thresholding
    return np.sign(W) * np.maximum(0.0, np.abs(W) - thresh)


# ============================================================
# Armijo line-search (com proximal L1)
# ============================================================

def armijo_search_alpha_epoch(
    W, b,
    X, y, classes,
    dW, db,
    l1, l2,
    alpha_init=1.0, beta=0.5, sigma=1e-4,
):
    """
    Busca alpha por backtracking (Armijo).
    Atualização:
      W_new = prox_{alpha*l1/m}(W - alpha*dW)
      b_new = b - alpha*db
    Condição Armijo baseada na perda com L1+L2.
    """
    m = X.shape[0]
    loss0, _ = softmax_loss(W, b, X, y, classes, l1=l1, l2=l2)

    alpha = float(alpha_init)
    max_tries = 50

    for _ in range(max_tries):
        W_try = prox_l1(W - alpha * dW, thresh=(alpha * l1) / m)
        b_try = b - alpha * db

        loss_try, _ = softmax_loss(W_try, b_try, X, y, classes, l1=l1, l2=l2)

        stepW = W_try - W
        stepb = b_try - b
        descent = float(np.sum(dW * stepW) + np.sum(db * stepb))

        if loss_try <= loss0 + sigma * descent:
            return alpha, W_try, b_try, loss_try

        alpha *= beta

    W_try = prox_l1(W - alpha * dW, thresh=(alpha * l1) / m)
    b_try = b - alpha * db
    loss_try, _ = softmax_loss(W_try, b_try, X, y, classes, l1=l1, l2=l2)
    return alpha, W_try, b_try, loss_try


def treinar_softmax_armijo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes_fixas: np.ndarray,
    l1: float,
    l2: float,
    epochs: int,
    alpha_init: float,
    beta: float,
    sigma: float,
    seed: int,
):
    rng = np.random.default_rng(seed)

    W = (0.01 * rng.standard_normal((X_train.shape[1], classes_fixas.shape[0]))).astype(np.float32)
    b = np.zeros((classes_fixas.shape[0],), dtype=np.float32)

    alphas = []

    for ep in range(1, epochs + 1):
        dW, db, _ = softmax_grad(W, b, X_train, y_train, classes_fixas, l2=l2)

        alpha, W_new, b_new, loss_new = armijo_search_alpha_epoch(
            W=W, b=b,
            X=X_train, y=y_train, classes=classes_fixas,
            dW=dW, db=db,
            l1=l1, l2=l2,
            alpha_init=alpha_init, beta=beta, sigma=sigma
        )
        alphas.append(alpha)
        W, b = W_new, b_new

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"  [Treino] ep={ep:03d}/{epochs} loss={loss_new:.6f} alpha={alpha:.3e}")

    return {"W": W, "b": b, "classes": classes_fixas, "alphas": np.array(alphas, dtype=np.float32)}


def predict_labels(X: np.ndarray, W: np.ndarray, b: np.ndarray, classes: np.ndarray):
    P = stable_softmax(X @ W + b)
    idx = np.argmax(P, axis=1)
    y_pred = classes[idx]
    return y_pred.astype(np.int64), P


# ============================================================
# Métricas / prints
# ============================================================

def report_accuracy(nome: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc = float(np.mean(y_true == y_pred))
    print(f"[{nome}] acc={acc:.4f} ({int(np.sum(y_true == y_pred))}/{len(y_true)})")
    return acc


def mostrar_exemplos_previsao(y_true, y_pred, P, n=10, seed=123):
    rng = np.random.default_rng(seed)
    n = min(n, len(y_true))
    idxs = rng.choice(len(y_true), size=n, replace=False)
    print("\n[Exemplos de previsão] (y_true -> y_pred | top3 prob)")
    for i in idxs:
        probs = P[i]
        top3 = np.argsort(-probs)[:3]
        top3_str = ", ".join([f"{int(top)}:{probs[top]:.3f}" for top in top3])
        ok = "OK" if y_true[i] == y_pred[i] else "ERR"
        print(f"  {ok}  {int(y_true[i])} -> {int(y_pred[i])} | {top3_str}")


def matriz_confusao_top_k(y_t, y_p, top_k=10):
    classes, counts = np.unique(y_t, return_counts=True)
    order = np.argsort(-counts)
    top = classes[order[:top_k]]
    labels = top
    cm = confusion_matrix(y_t, y_p, labels=labels)

    print(f"\n[Matriz de Confusão] Top-{top_k} classes mais comuns no TESTE")
    print("Labels (classe):", labels.tolist())
    print(cm)


# ============================================================
# CV: escolher lambdas (grid) via StratifiedKFold
# ============================================================

def selecionar_combos_l1_l2(grid_l1, grid_l2, max_combos: int = 8, strategy: str = "cover", seed: int = 42):
    g1 = sorted({float(v) for v in grid_l1})
    g2 = sorted({float(v) for v in grid_l2})

    if len(g1) == 0 or len(g2) == 0:
        return [(0.0, 0.0)]

    full = [(a, b) for a in g1 for b in g2]
    if max_combos is None or max_combos <= 0 or len(full) <= max_combos:
        return full

    rng = np.random.default_rng(seed)

    chosen = []
    chosen_set = set()

    def _add(pair):
        if pair not in chosen_set:
            chosen.append(pair)
            chosen_set.add(pair)

    if strategy == "cover":
        i0, im, i1 = 0, len(g1) // 2, len(g1) - 1
        j0, jm, j1 = 0, len(g2) // 2, len(g2) - 1
        base = [
            (g1[i0], g2[j0]), (g1[i0], g2[jm]), (g1[i0], g2[j1]),
            (g1[im], g2[j0]), (g1[im], g2[jm]), (g1[im], g2[j1]),
            (g1[i1], g2[j0]), (g1[i1], g2[j1]),
        ]
        for p in base:
            _add(p)

    remaining = [p for p in full if p not in chosen_set]
    rng.shuffle(remaining)
    for p in remaining:
        if len(chosen) >= max_combos:
            break
        _add(p)

    return chosen[:max_combos]


def escolher_melhores_lambdas_por_cv(
    X_train_feat: np.ndarray,
    y_train_labels: np.ndarray,
    classes_fixas: np.ndarray,
    k_folds: int,
    grid_l1,
    grid_l2,
    epochs: int,
    alpha_init: float,
    beta: float,
    sigma: float,
    seed: int,
    max_combos: int = 8,
    combo_strategy: str = "cover",
    seed_combos: int | None = None,
):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    if seed_combos is None:
        seed_combos = seed

    combos = selecionar_combos_l1_l2(
        grid_l1=grid_l1,
        grid_l2=grid_l2,
        max_combos=int(max_combos),
        strategy=str(combo_strategy),
        seed=int(seed_combos),
    )

    g1 = sorted({float(v) for v in grid_l1})
    g2 = sorted({float(v) for v in grid_l2})
    total_full = len(g1) * len(g2)
    print(f"[CV] Grid completo: {len(g1)}x{len(g2)}={total_full} | avaliando {len(combos)} combos")
    print("[CV] Combos (l1,l2):", combos)

    best = None
    best_score = -1.0
    best_alpha_med = None

    for (l1, l2) in combos:
        accs = []
        alphas_medianas = []

        print(f"\n[CV] Testando l1={l1} | l2={l2} ...")
        fold = 0
        for tr_idx, va_idx in skf.split(X_train_feat, y_train_labels):
            fold += 1
            X_tr, y_tr = X_train_feat[tr_idx], y_train_labels[tr_idx]
            X_va, y_va = X_train_feat[va_idx], y_train_labels[va_idx]

            modelo = treinar_softmax_armijo(
                X_tr, y_tr, classes_fixas=classes_fixas,
                l1=float(l1), l2=float(l2),
                epochs=int(epochs), alpha_init=alpha_init, beta=beta, sigma=sigma,
                seed=seed + fold,
            )
            y_pred_va, _ = predict_labels(X_va, modelo["W"], modelo["b"], modelo["classes"])
            acc = float(np.mean(y_pred_va == y_va))
            accs.append(acc)

            if modelo["alphas"].size > 0:
                alphas_medianas.append(float(np.median(modelo["alphas"])))

            print(f"  [CV fold {fold}/{k_folds}] acc={acc:.4f}")

        mean_acc = float(np.mean(accs))
        med_alpha = float(np.median(alphas_medianas)) if len(alphas_medianas) > 0 else float(alpha_init)
        print(f"[CV] l1={l1} l2={l2} -> mean_acc={mean_acc:.4f} | alpha_mediana~{med_alpha:.3e}")

        if mean_acc > best_score:
            best_score = mean_acc
            best = (float(l1), float(l2))
            best_alpha_med = med_alpha

    return best, best_score, best_alpha_med


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Dataset path: {CONFIG['dataset_path']}")
    print(f"Exists? {Path(CONFIG['dataset_path']).exists()}")

    X, y = carregar_joblib_dataset(CONFIG["dataset_path"])

    print("\n[Info] Dataset original")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("n classes:", len(np.unique(y)))

    diagnostico_hog_dim(X.shape[1])

    # 1) selecionar classes elegíveis e amostrar
    print(f"\n[Etapa 1/7] Selecionando {100*CONFIG['frac_classes']:.3f}% das classes ELEGÍVEIS "
          f"(seed={CONFIG['seed_classes']}) com mínimo {CONFIG['min_amostras_por_classe']} amostras/classe...")

    classes_sel, classes_elig = escolher_classes(
        y, frac_classes=CONFIG["frac_classes"], seed=CONFIG["seed_classes"],
        min_amostras=CONFIG["min_amostras_por_classe"]
    )
    print(f"[Seleção] Classes elegíveis (>= {CONFIG['min_amostras_por_classe']}): {len(classes_elig)} de {len(np.unique(y))}")

    Xf, yf = filtrar_por_classes(X, y, classes_sel)
    print(f"[Etapa 1/7] Após filtro:")
    print(f"  X: {Xf.shape} | n classes: {len(np.unique(yf))} | classes escolhidas: {len(classes_sel)}")

    if Xf.shape[0] == 0:
        raise RuntimeError("Seleção de classes resultou em dataset vazio. Ajuste frac_classes/min_amostras.")

    # 2) split treino/teste
    print(f"\n[Etapa 2/7] Split treino/teste (test_frac={CONFIG['test_frac']})...")
    X_train, X_test, y_train, y_test = train_test_split(
        Xf, yf, test_size=CONFIG["test_frac"], random_state=CONFIG["seed_split"], stratify=yf
    )
    print(f"[Etapa 2/7] Train: {X_train.shape} | Test: {X_test.shape}")

    # 3) garante mínimo no treino para manter classes
    min_train = max(CONFIG["min_train_por_classe"], CONFIG["k_folds"])
    print(f"\n[Etapa 3/7] Garantindo mínimo de {min_train} amostras por classe no TREINO (pré-CV)...")
    X_train, y_train, classes_train = filtrar_min_train_por_classe(X_train, y_train, min_train=min_train)
    print(f"[Etapa 3/7] Train após filtro: {X_train.shape} | n classes: {len(classes_train)}")

    # filtra o teste para manter mesmas classes do treino (evita classes inexistentes)
    mask_test = np.isin(y_test, classes_train)
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]
    print(f"[Etapa 3/7] Test após alinhar classes: {X_test.shape} | n classes: {len(np.unique(y_test))}")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Treino ficou com <2 classes após filtros. Ajuste frac_classes/min_train.")

    # 4) padronização (média/var do treino completo)
    print("\n[Etapa 4/7] Padronizando features (mean/std do TREINO)...")
    mu = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X_train.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, CONFIG["eps_std"])

    X_train_feat = ((X_train - mu) / sd).astype(np.float32)
    X_test_feat = ((X_test - mu) / sd).astype(np.float32)

    # 5) amostrar treino para CV (cv_frac), garantindo >= k_folds por classe
    print(f"\n[Etapa 5/7] Amostrando TREINO para CV: {100*CONFIG['cv_frac']:.2f}% (seed={CONFIG['seed_split']}) "
          f"com {CONFIG['k_folds']} por classe...")

    def amostrar_para_cv_por_classes(y_full: np.ndarray, frac: float, seed: int, min_por_classe: int):
        rng = np.random.default_rng(seed)
        classes = np.unique(y_full)

        idx_keep = []
        for c in classes:
            idx_c = np.flatnonzero(y_full == c)
            if len(idx_c) < min_por_classe:
                continue
            pick = rng.choice(idx_c, size=min_por_classe, replace=False)
            idx_keep.append(pick)

        if len(idx_keep) == 0:
            return np.array([], dtype=np.int64), classes

        idx_keep = np.concatenate(idx_keep).astype(np.int64)

        target = int(np.ceil(frac * len(y_full)))
        target = max(target, len(idx_keep))
        if target == len(idx_keep):
            return np.unique(idx_keep), classes

        remaining = np.setdiff1d(np.arange(len(y_full), dtype=np.int64), idx_keep, assume_unique=False)
        y_rem = y_full[remaining]

        add = target - len(idx_keep)
        if add <= 0:
            return np.unique(idx_keep), classes

        if len(remaining) <= add:
            idx_final = np.unique(np.concatenate([idx_keep, remaining]))
            return idx_final.astype(np.int64), classes

        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=add, random_state=seed)
            idx_add_local, _ = next(sss.split(np.zeros_like(y_rem), y_rem))
            idx_add = remaining[idx_add_local]
        except Exception:
            idx_add = rng.choice(remaining, size=add, replace=False)

        idx_final = np.unique(np.concatenate([idx_keep, idx_add]).astype(np.int64))
        return idx_final, classes

    idx_cv, _ = amostrar_para_cv_por_classes(
        y_train, frac=CONFIG["cv_frac"], seed=CONFIG["seed_split"], min_por_classe=CONFIG["k_folds"]
    )
    if idx_cv.size == 0:
        raise RuntimeError("Amostra para CV ficou vazia. Ajuste cv_frac/min_amostras/k_folds.")

    X_cv = X_train_feat[idx_cv]
    y_cv = y_train[idx_cv]
    print(f"[Etapa 5/7] CV sample: X={X_cv.shape} | classes={len(np.unique(y_cv))}")
    _, cts = np.unique(y_cv, return_counts=True)
    print(f"[Etapa 5/7] min count por classe no CV sample = {int(cts.min())} (deve ser >= {CONFIG['k_folds']})")

    # 6) escolher lambdas por CV (grid reduzido)
    epochs_cv = int(CONFIG.get("epochs_cv", CONFIG.get("epochs", 50)))
    print(f"\n[Etapa 6/7] Rodando grid-search por CV (com Armijo + L1/L2) | epochs_cv={epochs_cv}...")
    (best_l1, best_l2), best_score, best_alpha_med = escolher_melhores_lambdas_por_cv(
        X_train_feat=X_cv,
        y_train_labels=y_cv,
        classes_fixas=np.unique(y_cv).astype(np.int64),
        k_folds=CONFIG["k_folds"],
        grid_l1=CONFIG["grid_l1"],
        grid_l2=CONFIG["grid_l2"],
        epochs=epochs_cv,
        alpha_init=CONFIG["alpha_init"],
        beta=CONFIG["armijo_beta"],
        sigma=CONFIG["armijo_sigma"],
        seed=CONFIG["seed_split"],
        max_combos=CONFIG.get("max_combos_cv", 8),
        combo_strategy=CONFIG.get("combo_strategy_cv", "cover"),
        seed_combos=CONFIG.get("seed_combos_cv", CONFIG["seed_split"]),
    )
    print(f"\n[CV] Melhor: l1={best_l1} | l2={best_l2} | mean_acc={best_score:.4f} | alpha_mediana~{best_alpha_med:.3e}")

    # 7) treino final em amostra maior (final_frac) e avaliação no teste
    epochs_final = int(CONFIG.get("epochs_final", CONFIG.get("epochs", 80)))
    print(f"\n[Etapa 7/7] Treino final em {100*CONFIG['final_frac']:.1f}% do treino | epochs_final={epochs_final}...")
    idx_final = amostrar_estratificado(y_train, frac=CONFIG["final_frac"], seed=CONFIG["seed_split"])
    X_final = X_train_feat[idx_final]
    y_final = y_train[idx_final]
    classes_final = np.unique(y_final).astype(np.int64)

    print(f"[Etapa 7/7] Final sample: X={X_final.shape} | classes={len(classes_final)}")

    modelo_final = treinar_softmax_armijo(
        X_train=X_final,
        y_train=y_final,
        classes_fixas=classes_final,
        l1=float(best_l1), l2=float(best_l2),
        epochs=epochs_final,
        alpha_init=best_alpha_med if best_alpha_med is not None else CONFIG["alpha_init"],
        beta=CONFIG["armijo_beta"], sigma=CONFIG["armijo_sigma"],
        seed=CONFIG["seed_split"] + 999,
    )

    # avaliação treino (na amostra final) e teste
    y_pred_train, P_train = predict_labels(X_final, modelo_final["W"], modelo_final["b"], modelo_final["classes"])
    y_pred_test, P_test = predict_labels(X_test_feat, modelo_final["W"], modelo_final["b"], modelo_final["classes"])

    report_accuracy("TREINO(final sample)", y_final, y_pred_train)
    report_accuracy("TESTE", y_test, y_pred_test)

    mostrar_exemplos_previsao(y_test, y_pred_test, P_test, n=CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])
    matriz_confusao_top_k(y_test, y_pred_test, top_k=CONFIG["top_k_confusao"])


if __name__ == "__main__":
    main()
