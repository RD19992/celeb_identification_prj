import joblib
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import StratifiedKFold


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "dataset_filename": "celeba_hog_128x128_o9.joblib",

    # prototipagem / seleção de classes
    "frac_classes": 1.00,          # fração das CLASSES ELEGÍVEIS
    "seed_classes": 42,
    "min_amostras_por_classe": 25,  # mínimo no dataset inteiro p/ ser elegível

    # split
    "test_frac": 0.20,
    "seed_split": 42,
    "n_min_treino_por_classe": 25,

    # CV
    "k_folds": 5,
    "seed_cv": 42,

    # amostras separadas do TREINO
    "frac_amostras_cv": 0.01,       # 1% do TREINO para CV
    "frac_amostras_final": 1.00,    # 100% do TREINO para treino final
    "seed_amostras_cv": 123,
    "seed_amostras_final": 456,

    # grid lambdas
    "lambda_l1_grid": [0.0, 0.05, 0.1, 0.5],
    "lambda_l2_grid": [0.0, 0.05, 0.1, 0.5],

    # treino
    "epocas_cv": 15,
    "epocas_final": 300,

    "tamanho_lote": 256,
    "imprimir_a_cada_n_lotes": 25,

    # z-score
    "std_eps": 1e-8,

    # diagnóstico / logging
    "avaliar_loss_subamostra_por_epoca": True,
    "loss_subamostra_max": 5000,
    "diag_seed": 42,

    # output
    "n_exemplos_previsao": 10,

    # ========================================================
    # [ALTERAÇÃO] Armijo (backtracking) por ÉPOCA
    # - começa com alpha0 grande
    # - busca só no treino/CV (probe batch do treino)
    # ========================================================
    "use_armijo_no_cv": True,
    "use_armijo_no_final": True,

    "armijo_alpha0": 1.0,          # alpha inicial "grande"
    "armijo_beta": 0.5,            # fator de backtracking
    "armijo_c1": 1e-4,             # constante de Armijo
    "armijo_max_backtracks": 12,   # limite para não ficar caro
    "armijo_alpha_min": 1e-6,      # piso
    "armijo_growth": 1.25,         # warm-start: tenta aumentar um pouco o alpha a cada época (capado em alpha0)
    "armijo_probe_batch": 512,     # batch usado para a busca (treino apenas)
}


# ============================================================
# Diagnóstico HOG
# ============================================================

def diagnostico_dim_hog(d: int):
    print("\n[Diagnóstico HOG] Dimensão de features (d) =", d)
    if d == 1764:
        print("[Diagnóstico HOG] d=1764 sugere HOG 64x64 (o9, cell8, block2).")
        return
    if d == 8100:
        print("[Diagnóstico HOG] d=8100 sugere HOG 128x128 (o9, cell8, block2).")
        return
    est_cells_minus_1 = np.sqrt(max(d, 1) / 36.0)
    est_img = 8.0 * (est_cells_minus_1 + 1.0)
    print("[Diagnóstico HOG] d não bate com 64x64/128x128 típicos.")
    print(f"[Diagnóstico HOG] Estimativa grosseira de IMG_SIZE ≈ {est_img:.1f}px (assumindo o9, cell8, block2).")


# ============================================================
# Utilitários (labels, métricas)
# ============================================================

def codificar_rotulos_com_classes(y: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes, dtype=np.int64)
    mapa = {int(c): i for i, c in enumerate(classes.tolist())}
    y_idx = np.array([mapa[int(v)] for v in y], dtype=np.int64)
    return y_idx, classes, mapa

def calcular_erro_classificacao(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))

def mostrar_previsoes_amostrais(nome: str, y_true: np.ndarray, y_pred: np.ndarray, n_amostras: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(n_amostras, n), replace=False)
    print(f"\n[{nome}] Exemplos aleatórios (verdadeiro -> predito):")
    for t, i in enumerate(idx, start=1):
        print(f"  #{t:02d} | y_true={int(y_true[i])} -> y_pred={int(y_pred[i])}")


# ============================================================
# Padronização (z-score) - tudo float32
# ============================================================

def fit_standardizer(X_train: np.ndarray, eps: float):
    X_train = X_train.astype(np.float32, copy=False)
    mean = np.mean(X_train, axis=0, dtype=np.float32).astype(np.float32, copy=False)
    std = np.std(X_train, axis=0, dtype=np.float32).astype(np.float32, copy=False)
    std = np.maximum(std, np.float32(eps)).astype(np.float32, copy=False)
    return mean, std

def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X = X.astype(np.float32, copy=False)
    return (X - mean) / std


# ============================================================
# Seleção de classes: elegíveis primeiro
# ============================================================

def selecionar_classes_aleatorias_entre_elegiveis(X, y, frac_classes: float, seed: int, min_amostras_por_classe: int):
    rng = np.random.default_rng(seed)
    classes_all, counts_all = np.unique(y, return_counts=True)
    elegiveis = classes_all[counts_all >= min_amostras_por_classe]

    print(f"[Seleção] Classes elegíveis (>= {min_amostras_por_classe}): {len(elegiveis)} de {len(classes_all)}")

    if len(elegiveis) == 0:
        return X[:0], y[:0], elegiveis

    if frac_classes >= 1.0:
        escolhidas = elegiveis
    else:
        n_escolher = max(1, int(np.round(frac_classes * len(elegiveis))))
        n_escolher = min(n_escolher, len(elegiveis))
        escolhidas = rng.choice(elegiveis, size=n_escolher, replace=False)

    mask = np.isin(y, escolhidas)
    return X[mask], y[mask], np.asarray(escolhidas, dtype=y.dtype)


# ============================================================
# Split estratificado com mínimo no treino e (quando possível) 1+ no teste
# ============================================================

def split_estratificado_min_treino(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float,
    seed: int,
    min_treino_por_classe: int
):
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)

    order = np.argsort(y)
    y_sorted = y[order]
    _, start_idx = np.unique(y_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(y_sorted))

    train_idx = []
    test_idx = []

    for s, e in zip(start_idx, end_idx):
        grp = order[s:e]
        rng.shuffle(grp)
        n_i = len(grp)

        if n_i <= 1:
            train_idx.append(grp)
            continue

        min_tr = min(min_treino_por_classe, n_i - 1)
        test_target = int(np.round(test_frac * n_i))
        if test_target <= 0:
            test_target = 1
        max_test = n_i - min_tr
        test_take = min(test_target, max_test)

        if test_take <= 0:
            train_idx.append(grp)
        else:
            test_idx.append(grp[:test_take])
            train_idx.append(grp[test_take:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if len(test_idx) else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def filtrar_classes_min_train_para_cv(X_train, y_train, X_test, y_test, min_train_por_classe: int):
    classes, counts = np.unique(y_train, return_counts=True)
    classes_ok = classes[counts >= min_train_por_classe]
    mask_tr = np.isin(y_train, classes_ok)
    mask_te = np.isin(y_test, classes_ok)
    return X_train[mask_tr], y_train[mask_tr], X_test[mask_te], y_test[mask_te], classes_ok


# ============================================================
# Amostragem do TREINO (duas amostras):
#   - CV: escolhe classes e pega k_folds por classe até bater ~frac
#   - Final: pega fração por classe, restrita às classes do CV
# ============================================================

def amostrar_para_cv_por_classes(y_train: np.ndarray, frac: float, seed: int, min_por_classe: int):
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)

    N = int(len(y_train))
    alvo = max(min_por_classe, int(np.round(frac * N)))
    if alvo <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    classes, counts = np.unique(y_train, return_counts=True)
    classes_ok = classes[counts >= min_por_classe]
    if len(classes_ok) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng.shuffle(classes_ok)

    idx_por_classe = {}
    for c in classes_ok.tolist():
        idx = np.where(y_train == c)[0]
        rng.shuffle(idx)
        idx_por_classe[int(c)] = idx

    amostra = []
    classes_escolhidas = []
    for c in classes_ok.tolist():
        c = int(c)
        idx = idx_por_classe[c]
        if len(idx) < min_por_classe:
            continue
        amostra.append(idx[:min_por_classe])
        classes_escolhidas.append(c)
        if sum(len(a) for a in amostra) >= alvo:
            break

    if len(amostra) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    amostra_idx = np.concatenate(amostra).astype(np.int64, copy=False)
    if len(amostra_idx) > alvo:
        rng.shuffle(amostra_idx)
        amostra_idx = amostra_idx[:alvo]

    classes_escolhidas = np.array(classes_escolhidas, dtype=np.int64)
    return amostra_idx, classes_escolhidas


def amostrar_final_estratificado(y_train: np.ndarray, classes_permitidas: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    classes_permitidas = np.asarray(classes_permitidas, dtype=np.int64)

    amostra = []
    for c in classes_permitidas.tolist():
        idx = np.where(y_train == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        take = int(np.round(frac * len(idx)))
        take = max(1, min(take, len(idx)))
        amostra.append(idx[:take])

    if len(amostra) == 0:
        return np.array([], dtype=np.int64)

    amostra_idx = np.concatenate(amostra).astype(np.int64, copy=False)
    rng.shuffle(amostra_idx)
    return amostra_idx


# ============================================================
# Softmax + Weighted CE (SEM one-hot) + Elastic Net correto
# ============================================================

def compute_class_weights_from_yidx(y_idx: np.ndarray, K: int, eps: float = 1e-12) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=K).astype(np.float32, copy=False)
    counts_safe = np.maximum(counts, np.float32(1.0))
    total = np.sum(counts_safe).astype(np.float32, copy=False)
    w = total / (np.float32(K) * counts_safe + np.float32(eps))
    return w.astype(np.float32, copy=False)

def softmax_probs(logits: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z).astype(np.float32, copy=False)
    denom = np.sum(expz, axis=1, keepdims=True).astype(np.float32, copy=False) + np.float32(eps)
    return (expz / denom).astype(np.float32, copy=False)

def data_loss_wce(params: dict, X: np.ndarray, y_idx: np.ndarray, class_weights: np.ndarray, eps: float = 1e-12) -> float:
    W = params["W"]
    b = params["b"]
    logits = (X @ W.T + b).astype(np.float32, copy=False)
    P = softmax_probs(logits, eps=eps)
    p_true = P[np.arange(len(y_idx)), y_idx]
    w = class_weights[y_idx].astype(np.float32, copy=False)
    loss = -np.mean(w * np.log(p_true + np.float32(eps)), dtype=np.float32)
    return float(loss)

def reg_loss_elasticnet(params: dict, lambda_l1: float, lambda_l2: float, n_train_total: int) -> float:
    W = params["W"]
    n = np.float32(max(n_train_total, 1))
    reg = (np.float32(lambda_l1) * np.sum(np.abs(W), dtype=np.float32) +
           np.float32(0.5) * np.float32(lambda_l2) * np.sum(W * W, dtype=np.float32)) / n
    return float(reg)

def batch_grad_softmax_wce_l2(
    params: dict,
    Xb: np.ndarray,
    yb_idx: np.ndarray,
    class_weights: np.ndarray,
    lambda_l2: float,
    n_train_total: int,
    eps: float = 1e-12
):
    """
    Grads do termo suave: WCE + L2 (L1 via proximal fora).
    Truque padrão (sem one-hot):
      P[range(m), y] -= 1
    """
    W = params["W"]
    b = params["b"]
    m = int(Xb.shape[0])
    if m <= 0:
        return 0.0, {"W": np.zeros_like(W), "b": np.zeros_like(b)}

    logits = (Xb @ W.T + b).astype(np.float32, copy=False)
    P = softmax_probs(logits, eps=eps)

    w = class_weights[yb_idx].astype(np.float32, copy=False)
    p_true = P[np.arange(m), yb_idx]
    loss_data = -np.mean(w * np.log(p_true + np.float32(eps)), dtype=np.float32)

    dlogits = P
    dlogits[np.arange(m), yb_idx] -= np.float32(1.0)
    dlogits *= (w[:, None] / np.float32(m)).astype(np.float32, copy=False)

    grad_W = (dlogits.T @ Xb).astype(np.float32, copy=False)
    grad_b = np.sum(dlogits, axis=0, dtype=np.float32).astype(np.float32, copy=False)

    if lambda_l2 != 0.0:
        grad_W += (np.float32(lambda_l2) / np.float32(max(n_train_total, 1))) * W

    return float(loss_data), {"W": grad_W, "b": grad_b}

def proximal_l1(W: np.ndarray, shrink: float) -> np.ndarray:
    shrink = np.float32(shrink)
    return (np.sign(W) * np.maximum(np.abs(W) - shrink, np.float32(0.0))).astype(np.float32, copy=False)


# ============================================================
# [ALTERAÇÃO] Armijo (por ÉPOCA): busca alpha no treino/CV apenas
# ============================================================

def objective_batch_total(params: dict, Xb: np.ndarray, yb_idx: np.ndarray,
                          class_weights: np.ndarray, lambda_l1: float, lambda_l2: float, n_train_total: int) -> float:
    return float(
        data_loss_wce(params, Xb, yb_idx, class_weights) +
        reg_loss_elasticnet(params, lambda_l1, lambda_l2, n_train_total)
    )

def apply_prox_step(params: dict, grads: dict, alpha: float,
                    lambda_l1: float, n_train_total: int) -> dict:
    a = np.float32(alpha)
    W_new = (params["W"] - a * grads["W"]).astype(np.float32, copy=False)
    b_new = (params["b"] - a * grads["b"]).astype(np.float32, copy=False)
    if lambda_l1 != 0.0:
        shrink = a * (np.float32(lambda_l1) / np.float32(max(n_train_total, 1)))
        W_new = proximal_l1(W_new, shrink)
    return {"W": W_new, "b": b_new}

def armijo_search_alpha_epoch(params: dict, grads_probe: dict,
                             X_probe: np.ndarray, y_probe_idx: np.ndarray,
                             class_weights: np.ndarray,
                             lambda_l1: float, lambda_l2: float, n_train_total: int,
                             alpha_start: float,
                             alpha0: float, beta: float, c1: float,
                             max_backtracks: int, alpha_min: float) -> tuple[float, int, float, float]:
    """
    Busca alpha com Armijo em um probe batch DO TREINO:
      F(new) <= F(old) - c1 * alpha * ||grad||^2
    (pragmático p/ proximal gradient em mini-batch)
    """
    alpha = float(min(alpha0, max(alpha_min, alpha_start)))
    F_old = objective_batch_total(params, X_probe, y_probe_idx, class_weights, lambda_l1, lambda_l2, n_train_total)

    gW = grads_probe["W"]
    gb = grads_probe["b"]
    grad_norm_sq = float(np.sum(gW * gW, dtype=np.float32) + np.sum(gb * gb, dtype=np.float32))
    if grad_norm_sq <= 0.0:
        return alpha_min, 0, F_old, F_old

    bt = 0
    while True:
        params_try = apply_prox_step(params, grads_probe, alpha, lambda_l1, n_train_total)
        F_new = objective_batch_total(params_try, X_probe, y_probe_idx, class_weights, lambda_l1, lambda_l2, n_train_total)

        if (F_new <= F_old - c1 * alpha * grad_norm_sq) or (alpha <= alpha_min) or (bt >= max_backtracks):
            return float(max(alpha, alpha_min)), bt, F_old, F_new

        alpha *= beta
        bt += 1


# ============================================================
# Treino (SGD) usando alpha por época (Armijo ou fallback fixo)
# ============================================================

def treinar_softmax_elasticnet_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes_modelo: np.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    epocas: int,
    tamanho_lote: int,
    seed: int,
    imprimir_a_cada_n_lotes: int,
    avaliar_loss_subamostra_por_epoca: bool,
    loss_subamostra_max: int,
    use_armijo: bool,
):
    rng = np.random.default_rng(seed)

    # mapeia y para idx internos conforme classes_modelo
    y_idx, classes, mapa = codificar_rotulos_com_classes(y_train, classes_modelo)
    K = len(classes)
    d = int(X_train.shape[1])
    n_train_total = int(X_train.shape[0])

    class_weights = compute_class_weights_from_yidx(y_idx, K)

    W = rng.normal(0.0, 0.01, size=(K, d)).astype(np.float32)
    b = np.zeros(K, dtype=np.float32)
    params = {"W": W, "b": b}

    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n_train_total:
        tamanho_lote = n_train_total
    n_lotes = int(np.ceil(n_train_total / tamanho_lote))

    hist = []
    alphas_epoch = []
    backtracks_epoch = []

    # warm-start do alpha (começa grande)
    alpha_prev = float(CONFIG["armijo_alpha0"])

    for epoca in range(epocas):
        # ----------------------------------------------------
        # [ALTERAÇÃO] define alpha da ÉPOCA por Armijo no TREINO
        # ----------------------------------------------------
        if use_armijo:
            probe_n = min(int(CONFIG["armijo_probe_batch"]), n_train_total)
            probe_idx = rng.choice(n_train_total, size=probe_n, replace=False)
            Xp = X_train[probe_idx].astype(np.float32, copy=False)
            yp = y_idx[probe_idx]

            # grads no probe batch (termo suave: WCE + L2)
            _, grads_probe = batch_grad_softmax_wce_l2(
                params, Xp, yp, class_weights,
                lambda_l2=lambda_l2,
                n_train_total=n_train_total
            )

            # tenta crescer um pouco a cada época, capado em alpha0
            alpha_start = min(CONFIG["armijo_alpha0"], alpha_prev * float(CONFIG["armijo_growth"]))
            alpha_ep, bt, F_old, F_new = armijo_search_alpha_epoch(
                params=params,
                grads_probe=grads_probe,
                X_probe=Xp,
                y_probe_idx=yp,
                class_weights=class_weights,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                n_train_total=n_train_total,
                alpha_start=alpha_start,
                alpha0=float(CONFIG["armijo_alpha0"]),
                beta=float(CONFIG["armijo_beta"]),
                c1=float(CONFIG["armijo_c1"]),
                max_backtracks=int(CONFIG["armijo_max_backtracks"]),
                alpha_min=float(CONFIG["armijo_alpha_min"])
            )

            alpha_prev = float(alpha_ep)
            alphas_epoch.append(float(alpha_ep))
            backtracks_epoch.append(int(bt))

            print(f"[Armijo] Época {epoca+1}/{epocas} | alpha={alpha_ep:.3e} | backtracks={bt} | Fprobe {F_old:.4f}->{F_new:.4f}")
        else:
            # fallback: alpha fixo (não usado na sua solicitação atual)
            alpha_ep = 0.02
            alphas_epoch.append(float(alpha_ep))
            backtracks_epoch.append(0)

        a32 = np.float32(alpha_ep)

        # ----------------------------------------------------
        # SGD com alpha fixo nessa época
        # ----------------------------------------------------
        idx = rng.permutation(n_train_total)
        for lote_i, ini in enumerate(range(0, n_train_total, tamanho_lote), start=1):
            batch = idx[ini:ini + tamanho_lote]
            Xb = X_train[batch].astype(np.float32, copy=False)
            yb = y_idx[batch]

            loss_b, grads = batch_grad_softmax_wce_l2(
                params, Xb, yb, class_weights,
                lambda_l2=lambda_l2,
                n_train_total=n_train_total
            )

            params["W"] -= a32 * grads["W"]
            params["b"] -= a32 * grads["b"]

            if lambda_l1 != 0.0:
                shrink = a32 * (np.float32(lambda_l1) / np.float32(max(n_train_total, 1)))
                params["W"] = proximal_l1(params["W"], shrink)

            if (lote_i % imprimir_a_cada_n_lotes == 0) or (lote_i == n_lotes):
                print(
                    f"[Treino] Época {epoca+1}/{epocas} | Lote {lote_i}/{n_lotes} | "
                    f"alpha={alpha_ep:.3e} | loss_batch(data)={loss_b:.4f}",
                    end="\r"
                )

        print(" " * 140, end="\r")

        if avaliar_loss_subamostra_por_epoca:
            m = min(n_train_total, int(loss_subamostra_max))
            sub = rng.choice(n_train_total, size=m, replace=False)
            loss_data = data_loss_wce(params, X_train[sub].astype(np.float32, copy=False), y_idx[sub], class_weights)
            loss_reg = reg_loss_elasticnet(params, lambda_l1, lambda_l2, n_train_total)
            loss_total = loss_data + loss_reg
            hist.append(np.float32(loss_total))
            print(f"[SGD] Época {epoca+1}/{epocas} | alpha={alpha_ep:.3e} | loss~(data+reg)={loss_total:.6f}")

    stats = {
        "alpha_epoch": alphas_epoch,
        "alpha_mean": float(np.mean(alphas_epoch)) if len(alphas_epoch) else None,
        "alpha_median": float(np.median(alphas_epoch)) if len(alphas_epoch) else None,
        "alpha_min": float(np.min(alphas_epoch)) if len(alphas_epoch) else None,
        "alpha_max": float(np.max(alphas_epoch)) if len(alphas_epoch) else None,
        "armijo_backtracks_mean": float(np.mean(backtracks_epoch)) if len(backtracks_epoch) else None,
        "armijo_backtracks_max": int(np.max(backtracks_epoch)) if len(backtracks_epoch) else 0,
    }

    modelo = {
        "W": params["W"],
        "b": params["b"],
        "classes": classes,
        "mapa": mapa,  # debug
        "class_weights": class_weights,
        "lambda_l1": float(lambda_l1),
        "lambda_l2": float(lambda_l2),
        "n_train_total": int(n_train_total),
        "hist_loss": hist,
        "stats": stats,
    }
    return modelo


def prever(modelo: dict, X: np.ndarray):
    W = modelo["W"]
    b = modelo["b"]
    classes = modelo["classes"]
    logits = (X.astype(np.float32, copy=False) @ W.T + b).astype(np.float32, copy=False)
    pred_idx = np.argmax(logits, axis=1)
    return classes[pred_idx], logits


# ============================================================
# [ALTERAÇÃO] Avaliação com debug: por label e por idx interno
# ============================================================

def avaliar_com_debug(nome: str, modelo: dict, X: np.ndarray, y_true_labels: np.ndarray):
    y_true_labels = y_true_labels.astype(np.int64, copy=False)

    # 1) via labels
    y_pred_labels, logits = prever(modelo, X)
    y_pred_labels = y_pred_labels.astype(np.int64, copy=False)
    acc_labels = float(np.mean(y_true_labels == y_pred_labels))
    err_labels = 1.0 - acc_labels

    # 2) via idx interno (para garantir que mapeamento/classe não está bugado)
    classes = modelo["classes"].astype(np.int64, copy=False)
    y_true_idx, _, _ = codificar_rotulos_com_classes(y_true_labels, classes)
    pred_idx = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    acc_idx = float(np.mean(y_true_idx == pred_idx))
    err_idx = 1.0 - acc_idx

    print(f"\n[{nome}] Erro={err_labels:.4f} | Acurácia={acc_labels:.4f} | N={len(y_true_labels)}")
    print(f"[{nome}] Debug idx-interno: Erro={err_idx:.4f} | Acurácia={acc_idx:.4f}  (deve bater com acima)")

    if abs(acc_labels - acc_idx) > 1e-6:
        print(f"[{nome}] ALERTA: acc por label != acc por idx. Há bug de mapeamento/consistência!")

    return y_pred_labels


# ============================================================
# CV estratificado (SEM vazamento de scaler) + coleta alpha
# ============================================================

def escolher_melhores_lambdas_por_cv(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    lambda_l1_grid,
    lambda_l2_grid,
    k_folds: int,
    seed_cv: int,
):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_cv)
    combos = list(product(lambda_l1_grid, lambda_l2_grid))
    melhor = None

    total = len(combos)
    print(f"[CV] Grid-search: {total} combinações | {k_folds}-fold estratificado")

    for c_idx, (l1, l2) in enumerate(combos, start=1):
        erros = []
        alphas_med = []
        print(f"\n[CV] Combo {c_idx}/{total}: l1={l1} | l2={l2}")

        for f_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y_train), start=1):
            X_tr_raw, y_tr = X_train_raw[tr_idx], y_train[tr_idx]
            X_va_raw, y_va = X_train_raw[va_idx], y_train[va_idx]

            # classes do fold (robustez)
            classes_fold = np.unique(y_tr).astype(np.int64, copy=False)

            # scaler do fold (sem vazamento)
            mean_f, std_f = fit_standardizer(X_tr_raw, eps=CONFIG["std_eps"])
            X_tr = apply_standardizer(X_tr_raw, mean_f, std_f)
            X_va = apply_standardizer(X_va_raw, mean_f, std_f)

            print(f"[CV]  Fold {f_idx}/{k_folds} - treinando (Armijo={CONFIG['use_armijo_no_cv']})...")

            modelo = treinar_softmax_elasticnet_sgd(
                X_tr, y_tr, classes_fold,
                lambda_l1=l1, lambda_l2=l2,
                epocas=CONFIG["epocas_cv"],
                tamanho_lote=CONFIG["tamanho_lote"],
                seed=123 + 1000 * c_idx + f_idx,
                imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"],
                avaliar_loss_subamostra_por_epoca=False,
                loss_subamostra_max=0,
                use_armijo=bool(CONFIG["use_armijo_no_cv"]),
            )

            # erro no val
            y_pred_va, _ = prever(modelo, X_va)
            err = calcular_erro_classificacao(y_va.astype(np.int64), y_pred_va.astype(np.int64))
            erros.append(err)

            amed = modelo["stats"]["alpha_median"]
            if amed is not None:
                alphas_med.append(float(amed))

            print(f"[CV]  Fold {f_idx}/{k_folds} - erro={err:.4f} | alpha_median={amed}")

        mean_err = float(np.mean(erros))
        mean_alpha_med = float(np.mean(alphas_med)) if len(alphas_med) else None
        print(f"[CV] Média erro = {mean_err:.4f} | média(alpha_median)={mean_alpha_med}")

        if (melhor is None) or (mean_err < melhor[0]):
            melhor = (mean_err, l1, l2, mean_alpha_med)
            print(f"[CV] ** Novo melhor: erro={mean_err:.4f} | l1={l1} | l2={l2} | alpha_med~={mean_alpha_med}")

    return melhor


# ============================================================
# Avaliação no teste: top-10 confusão + métricas (igual ao seu)
# ============================================================

def _fmt_int(x: int, w: int) -> str:
    return str(int(x)).rjust(w)

def _fmt_float(x: float, w: int, d: int = 3) -> str:
    return f"{x:.{d}f}".rjust(w)

def imprimir_matriz_confusao(cm: np.ndarray, row_labels, col_labels, titulo: str, normalize_rows: bool = True):
    R, C = cm.shape
    row_labels = [str(x) for x in row_labels]
    col_labels = [str(x) for x in col_labels]

    w_label = max(6, max(len(s) for s in row_labels))
    w_cell = max(5, max(len(str(int(cm.max()))), max(len(s) for s in col_labels)))

    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)

    header = " " * (w_label + 2) + " ".join(s.rjust(w_cell) for s in col_labels) + " | " + "sum".rjust(w_cell)
    print(header)
    print("-" * len(header))

    for i in range(R):
        row = cm[i]
        s = int(row.sum())
        line = row_labels[i].rjust(w_label) + "  " + " ".join(_fmt_int(v, w_cell) for v in row) + " | " + _fmt_int(s, w_cell)
        print(line)

        if normalize_rows and s > 0:
            frac = (row.astype(np.float32) / np.float32(s)).astype(np.float32, copy=False)
            line2 = (" " * w_label) + "  " + " ".join(_fmt_float(float(v), w_cell, d=3) for v in frac) + " | " + _fmt_float(1.0, w_cell, d=3)
            print(line2)

    print("=" * 72 + "\n")

def avaliar_teste_top10(y_test: np.ndarray, y_pred_test: np.ndarray, top_k: int = 10):
    y_test = y_test.astype(np.int64, copy=False)
    y_pred_test = y_pred_test.astype(np.int64, copy=False)

    n = len(y_test)
    if n == 0:
        print("\n[Teste] Conjunto de teste vazio — nada para avaliar.")
        return

    acc = float(np.mean(y_test == y_pred_test))
    err = 1.0 - acc
    print(f"\n[Resultado TESTE] Erro: {err:.4f} | Acurácia: {acc:.4f} | N={n}")

    classes, counts = np.unique(y_test, return_counts=True)
    order = np.argsort(counts)[::-1]
    k = min(top_k, len(classes))
    top_classes = classes[order[:k]]
    top_counts = counts[order[:k]]

    print(f"\n[Teste] Top-{k} classes mais comuns (classe, count, fração no teste):")
    for c, cnt in zip(top_classes.tolist(), top_counts.tolist()):
        print(f"  {int(c)} | {int(cnt)} | {cnt / n:.4f}")

    mapa = {int(c): i for i, c in enumerate(top_classes.tolist())}
    OUT = k
    cm = np.zeros((k, k + 1), dtype=np.int64)

    for yt, yp in zip(y_test, y_pred_test):
        yt_i = mapa.get(int(yt), None)
        if yt_i is None:
            continue
        yp_j = mapa.get(int(yp), OUT)
        cm[yt_i, yp_j] += 1

    col_labels = [int(c) for c in top_classes.tolist()] + ["OUT"]
    row_labels = [int(c) for c in top_classes.tolist()]

    imprimir_matriz_confusao(
        cm,
        row_labels=row_labels,
        col_labels=col_labels,
        titulo=f"[Teste] Matriz de confusão (verdadeiro ∈ top-{k}; col=top-{k}+OUT). Linhas normalizadas abaixo.",
        normalize_rows=True
    )


# ============================================================
# MAIN
# ============================================================

def main():
    DATA_DIR = Path(__file__).resolve().parent
    DATASET_PATH = DATA_DIR / CONFIG["dataset_filename"]

    print("Dataset path:", DATASET_PATH)
    print("Exists?", DATASET_PATH.exists())

    data = joblib.load(DATASET_PATH)
    X, y = data["X"], data["y"]

    # float32 sempre
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print("\n[Info] Dataset original")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("n classes:", len(np.unique(y)))

    diagnostico_dim_hog(int(X.shape[1]))

    # Seleção de classes
    print(
        f"\n[Etapa 1/7] Selecionando {CONFIG['frac_classes']*100:.3f}% das classes ELEGÍVEIS (seed={CONFIG['seed_classes']}) "
        f"com mínimo {CONFIG['min_amostras_por_classe']} amostras/classe..."
    )
    X, y, escolhidas = selecionar_classes_aleatorias_entre_elegiveis(
        X, y,
        frac_classes=CONFIG["frac_classes"],
        seed=CONFIG["seed_classes"],
        min_amostras_por_classe=CONFIG["min_amostras_por_classe"]
    )

    if X.shape[0] == 0:
        print("\n[ERRO] Após filtro, não sobrou nenhuma amostra.")
        return

    print("[Etapa 1/7] Após filtro:")
    print("  X:", X.shape, "| n classes:", len(np.unique(y)), "| classes escolhidas:", len(escolhidas))

    # Split
    print(
        f"\n[Etapa 2/7] Split treino/teste (test_frac={CONFIG['test_frac']:.2f}) "
        f"com mínimo {CONFIG['n_min_treino_por_classe']} no treino por classe..."
    )
    X_train_all, X_test_all, y_train_all, y_test_all = split_estratificado_min_treino(
        X, y,
        test_frac=CONFIG["test_frac"],
        seed=CONFIG["seed_split"],
        min_treino_por_classe=CONFIG["n_min_treino_por_classe"]
    )

    # filtro pré-CV
    print(f"\n[Etapa 3/7] Garantindo mínimo de {CONFIG['k_folds']} amostras por classe no TREINO (pré-CV)...")
    X_train_all, y_train_all, X_test_all, y_test_all, _ = filtrar_classes_min_train_para_cv(
        X_train_all, y_train_all, X_test_all, y_test_all, min_train_por_classe=CONFIG["k_folds"]
    )
    if X_train_all.shape[0] == 0:
        print("\n[ERRO] Após pré-CV, treino vazio.")
        return

    print("[Etapa 3/7] Shapes após filtro pré-CV:")
    print("  Train(all):", X_train_all.shape, y_train_all.shape, "| n classes:", len(np.unique(y_train_all)))
    print("  Test (all):", X_test_all.shape, y_test_all.shape, "| n classes:", len(np.unique(y_test_all)))

    # Amostra CV
    print(
        f"\n[Etapa 4/7] Amostrando TREINO para CV: {CONFIG['frac_amostras_cv']*100:.2f}% "
        f"(seed={CONFIG['seed_amostras_cv']}) com {CONFIG['k_folds']} por classe..."
    )
    idx_cv, _ = amostrar_para_cv_por_classes(
        y_train_all,
        frac=CONFIG["frac_amostras_cv"],
        seed=CONFIG["seed_amostras_cv"],
        min_por_classe=CONFIG["k_folds"]
    )
    if len(idx_cv) == 0:
        print("\n[ERRO] CV sample vazio.")
        return

    X_train_cv_raw = X_train_all[idx_cv]
    y_train_cv = y_train_all[idx_cv]
    classes_cv = np.unique(y_train_cv).astype(np.int64, copy=False)

    print("[Etapa 4/7] CV sample:")
    print("  X_train_cv:", X_train_cv_raw.shape, "| n classes:", len(classes_cv))

    # Treino final: pool restrito às classes do CV, e amostra final
    mask_final_pool = np.isin(y_train_all, classes_cv)
    X_train_pool = X_train_all[mask_final_pool]
    y_train_pool = y_train_all[mask_final_pool]

    print(
        f"\n[Etapa 5/7] Amostrando TREINO para treino final: {CONFIG['frac_amostras_final']*100:.2f}% "
        f"(seed={CONFIG['seed_amostras_final']}) restrito às classes do CV..."
    )
    idx_final = amostrar_final_estratificado(
        y_train_pool,
        classes_permitidas=classes_cv,
        frac=CONFIG["frac_amostras_final"],
        seed=CONFIG["seed_amostras_final"]
    )
    if len(idx_final) == 0:
        print("\n[ERRO] Treino final sample vazio.")
        return

    X_train_final_raw = X_train_pool[idx_final]
    y_train_final = y_train_pool[idx_final]
    classes_final = np.unique(y_train_final).astype(np.int64, copy=False)  # robustez

    print("[Etapa 5/7] Final train sample:")
    print("  X_train_final:", X_train_final_raw.shape, "| n classes:", len(classes_final))

    # Teste restrito às classes do modelo final
    mask_test = np.isin(y_test_all, classes_final)
    X_test = X_test_all[mask_test]
    y_test = y_test_all[mask_test]
    print("\n[Info] Teste restrito às classes do modelo final:")
    print("  X_test:", X_test.shape, "| n classes:", len(np.unique(y_test)))

    # CV para lambdas (+ coleta alpha stats)
    print(f"\n[Etapa 6/7] CV estratificado (k={CONFIG['k_folds']}) para escolher lambdas (Armijo no treino do fold)...")
    best = escolher_melhores_lambdas_por_cv(
        X_train_raw=X_train_cv_raw,
        y_train=y_train_cv,
        lambda_l1_grid=CONFIG["lambda_l1_grid"],
        lambda_l2_grid=CONFIG["lambda_l2_grid"],
        k_folds=CONFIG["k_folds"],
        seed_cv=CONFIG["seed_cv"],
    )
    best_mean_err, best_l1, best_l2, best_alpha_med = best
    print(f"\n[CV] Melhor: mean_err={best_mean_err:.4f} | l1={best_l1} | l2={best_l2} | alpha_median~={best_alpha_med}")

    # scaler final
    print("\n[Etapa 7/7] Padronizando features (z-score) com stats do TREINO FINAL...")
    mean_tr, std_tr = fit_standardizer(X_train_final_raw, eps=CONFIG["std_eps"])
    X_train_feat = apply_standardizer(X_train_final_raw, mean_tr, std_tr)
    X_test_feat = apply_standardizer(X_test, mean_tr, std_tr)

    # treino final com Armijo (busca só no treino)
    print(f"\n[Treino FINAL] Treinando modelo final (Armijo={CONFIG['use_armijo_no_final']}, epocas={CONFIG['epocas_final']})...")
    modelo_final = treinar_softmax_elasticnet_sgd(
        X_train_feat, y_train_final, classes_final,
        lambda_l1=best_l1, lambda_l2=best_l2,
        epocas=CONFIG["epocas_final"],
        tamanho_lote=CONFIG["tamanho_lote"],
        seed=999,
        imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"],
        avaliar_loss_subamostra_por_epoca=CONFIG["avaliar_loss_subamostra_por_epoca"],
        loss_subamostra_max=CONFIG["loss_subamostra_max"],
        use_armijo=bool(CONFIG["use_armijo_no_final"]),
    )

    print("\n[Resumo Armijo FINAL]")
    st = modelo_final["stats"]
    print(f"  alpha_mean={st['alpha_mean']:.3e} | alpha_median={st['alpha_median']:.3e} | "
          f"alpha_min={st['alpha_min']:.3e} | alpha_max={st['alpha_max']:.3e}")
    print(f"  backtracks_mean={st['armijo_backtracks_mean']:.2f} | backtracks_max={st['armijo_backtracks_max']}")

    # ========================================================
    # [ALTERAÇÃO] Avaliar TREINO e TESTE com debug e exemplos
    # ========================================================
    y_pred_train = avaliar_com_debug("TREINO (final sample)", modelo_final, X_train_feat, y_train_final)
    y_pred_test = avaliar_com_debug("TESTE", modelo_final, X_test_feat, y_test)

    # exemplos explícitos (sem confusão)
    mostrar_previsoes_amostrais("TREINO", y_train_final, y_pred_train, CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])
    mostrar_previsoes_amostrais("TESTE", y_test, y_pred_test, CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])

    # avaliação top-10 no teste
    print("\n[Etapa Extra] Avaliando no TESTE (acurácia + matriz de confusão top-10)...")
    avaliar_teste_top10(y_test=y_test, y_pred_test=y_pred_test, top_k=10)


if __name__ == "__main__":
    main()
