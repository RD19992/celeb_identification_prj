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
    "frac_classes": 0.005,          # fração das CLASSES ELEGÍVEIS (ex.: 0.005 = 0.5%)
    "seed_classes": 42,
    "min_amostras_por_classe": 25,  # mínimo no dataset inteiro p/ ser elegível

    # split
    "test_frac": 0.20,
    "seed_split": 42,
    # recomendo ser < min_amostras_por_classe pra sobrar teste até quando count==min
    "n_min_treino_por_classe": 20,

    # CV
    "k_folds": 3,
    "seed_cv": 42,

    # grid lambdas
    "lambda_l1_grid": [0.0, 0.05, 0.1, 0.5],
    "lambda_l2_grid": [0.0, 0.05, 0.1, 0.5],

    # treino
    "epocas_cv": 15,
    "epocas_final": 30,

    # ========================================================
    # SGD (sem Armijo): LR fixo + decaimento por época
    # lr_epoch = max(lr_min, lr_inicial * (lr_decay ** epoch))
    # ========================================================
    "lr_inicial": 0.05,   # <- seguro; evite 0.5
    "lr_decay": 0.95,
    "lr_min": 1e-4,

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
    "exemplos_de": "teste",  # "treino" ou "teste"
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

def mostrar_previsoes_amostrais(y_true: np.ndarray, y_pred: np.ndarray, n_amostras: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(n_amostras, n), replace=False)
    print("\nExemplos aleatórios (verdadeiro -> predito):")
    for t, i in enumerate(idx, start=1):
        print(f"  #{t:02d} | y_true={int(y_true[i])} -> y_pred={int(y_pred[i])}")


# ============================================================
# Padronização (z-score)
# ============================================================

def fit_standardizer(X_train: np.ndarray, eps: float):
    X_train = X_train.astype(np.float32, copy=False)
    mean = np.mean(X_train, axis=0).astype(np.float32, copy=False)
    std = np.std(X_train, axis=0).astype(np.float32, copy=False)
    std = np.maximum(std, eps).astype(np.float32, copy=False)
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

        # garante min no treino, mas não pode consumir tudo
        min_tr = min(min_treino_por_classe, n_i - 1)
        # define teste alvo, garantindo 1 se sobrar
        test_target = int(np.round(test_frac * n_i))
        if test_target <= 0:
            test_target = 1
        # garante que sobra >= min_tr no treino
        max_test = n_i - min_tr
        test_take = min(test_target, max_test)

        # Se por algum motivo max_test==0, vai tudo pro treino
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
# Softmax + Weighted CE (SEM one-hot) + Elastic Net correto p/ SGD
#   - L2: entra no gradiente
#   - L1: proximal step (soft-threshold)
#   - Regularização escalada por n_train_total (não depende do batch)
# ============================================================

def compute_class_weights_from_yidx(y_idx: np.ndarray, K: int, eps: float = 1e-12) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=K).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    # peso inversamente proporcional à frequência (média ~1)
    w = total / (K * counts_safe + eps)
    return w.astype(np.float32)

def softmax_probs(logits: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z).astype(np.float32, copy=False)
    return expz / (np.sum(expz, axis=1, keepdims=True) + eps)

def data_loss_wce(params: dict, X: np.ndarray, y_idx: np.ndarray, class_weights: np.ndarray, eps: float = 1e-12) -> float:
    W = params["W"]
    b = params["b"]
    logits = X @ W.T + b
    P = softmax_probs(logits, eps=eps)
    p_true = P[np.arange(len(y_idx)), y_idx]
    w = class_weights[y_idx].astype(np.float32, copy=False)
    return float(-np.mean(w.astype(np.float64) * np.log(p_true + eps)))

def reg_loss_elasticnet(params: dict, lambda_l1: float, lambda_l2: float, n_train_total: int) -> float:
    W = params["W"]
    # escala por n_train_total (fixa)
    reg = (lambda_l1 * float(np.sum(np.abs(W))) + 0.5 * lambda_l2 * float(np.sum(W * W))) / float(max(n_train_total, 1))
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
    Retorna grads (W,b) para o termo de dados (WCE) + L2.
    L1 NÃO entra aqui (vamos aplicar proximal depois).
    """
    W = params["W"]
    b = params["b"]
    m = Xb.shape[0]
    if m <= 0:
        return 0.0, {"W": np.zeros_like(W), "b": np.zeros_like(b)}

    logits = Xb @ W.T + b
    P = softmax_probs(logits, eps=eps)  # (m,K)

    w = class_weights[yb_idx].astype(np.float32, copy=False)  # (m,)
    p_true = P[np.arange(m), yb_idx]
    loss_data = float(-np.mean(w.astype(np.float64) * np.log(p_true + eps)))

    # truque padrão
    dlogits = P
    dlogits[np.arange(m), yb_idx] -= 1.0
    dlogits *= (w[:, None] / float(m)).astype(np.float32, copy=False)

    grad_W = dlogits.T @ Xb   # (K,d)
    grad_b = np.sum(dlogits, axis=0)  # (K,)

    # L2 escalada por n_train_total (fixa, não depende do batch)
    if lambda_l2 != 0.0:
        grad_W += (lambda_l2 / float(max(n_train_total, 1))) * W

    return loss_data, {"W": grad_W.astype(np.float32, copy=False), "b": grad_b.astype(np.float32, copy=False)}

def proximal_l1(W: np.ndarray, shrink: float) -> np.ndarray:
    # soft-threshold
    return np.sign(W) * np.maximum(np.abs(W) - shrink, 0.0).astype(np.float32, copy=False)


# ============================================================
# SGD com LR fixo + decaimento
# ============================================================

def treinar_softmax_elasticnet_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes_fixas: np.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    lr_inicial: float,
    lr_decay: float,
    lr_min: float,
    epocas: int,
    tamanho_lote: int,
    seed: int,
    imprimir_a_cada_n_lotes: int,
    avaliar_loss_subamostra_por_epoca: bool,
    loss_subamostra_max: int,
):
    rng = np.random.default_rng(seed)

    y_idx, classes, _ = codificar_rotulos_com_classes(y_train, classes_fixas)
    K = len(classes)
    d = X_train.shape[1]
    n_train_total = int(X_train.shape[0])

    class_weights = compute_class_weights_from_yidx(y_idx, K)

    # init
    W = rng.normal(0.0, 0.01, size=(K, d)).astype(np.float32)
    b = np.zeros(K, dtype=np.float32)
    params = {"W": W, "b": b}

    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n_train_total:
        tamanho_lote = n_train_total
    n_lotes = int(np.ceil(n_train_total / tamanho_lote))

    hist = []
    stats = {"updates_total": 0, "lr0": float(lr_inicial), "decay": float(lr_decay), "lr_min": float(lr_min)}

    for epoca in range(epocas):
        lr = max(lr_min, lr_inicial * (lr_decay ** epoca))
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

            # grad step
            params["W"] -= lr * grads["W"]
            params["b"] -= lr * grads["b"]

            # proximal L1 (escala por n_train_total)
            if lambda_l1 != 0.0:
                shrink = lr * (lambda_l1 / float(max(n_train_total, 1)))
                params["W"] = proximal_l1(params["W"], shrink)

            stats["updates_total"] += 1

            if (lote_i % imprimir_a_cada_n_lotes == 0) or (lote_i == n_lotes):
                print(
                    f"[Treino] Época {epoca+1}/{epocas} | Lote {lote_i}/{n_lotes} | "
                    f"lr={lr:.3e} | loss_batch(data)={loss_b:.4f}",
                    end="\r"
                )

        print(" " * 120, end="\r")

        # loss por época (subamostra) + reg (computado 1x por época)
        if avaliar_loss_subamostra_por_epoca:
            m = min(n_train_total, int(loss_subamostra_max))
            sub = rng.choice(n_train_total, size=m, replace=False)
            loss_data = data_loss_wce(params, X_train[sub].astype(np.float32, copy=False), y_idx[sub], class_weights)
            loss_reg = reg_loss_elasticnet(params, lambda_l1, lambda_l2, n_train_total)
            loss_total = loss_data + loss_reg
            hist.append(loss_total)
            print(f"[SGD] Época {epoca+1}/{epocas} | lr={lr:.3e} | loss~(data+reg)={loss_total:.6f}")

    modelo = {
        "W": params["W"],
        "b": params["b"],
        "classes": classes,
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
    logits = X.astype(np.float32, copy=False) @ W.T + b
    pred_idx = np.argmax(logits, axis=1)
    return classes[pred_idx], logits


# ============================================================
# CV estratificado (SEM vazamento de scaler)
# ============================================================

def escolher_melhores_lambdas_por_cv(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    classes_fixas: np.ndarray,
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
        print(f"\n[CV] Combo {c_idx}/{total}: l1={l1} | l2={l2}")

        for f_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y_train), start=1):
            X_tr_raw, y_tr = X_train_raw[tr_idx], y_train[tr_idx]
            X_va_raw, y_va = X_train_raw[va_idx], y_train[va_idx]

            # scaler do fold (sem vazamento)
            mean_f, std_f = fit_standardizer(X_tr_raw, eps=CONFIG["std_eps"])
            X_tr = apply_standardizer(X_tr_raw, mean_f, std_f)
            X_va = apply_standardizer(X_va_raw, mean_f, std_f)

            print(f"[CV]  Fold {f_idx}/{k_folds} - treinando...")

            modelo = treinar_softmax_elasticnet_sgd(
                X_tr, y_tr, classes_fixas,
                lambda_l1=l1, lambda_l2=l2,
                lr_inicial=CONFIG["lr_inicial"],
                lr_decay=CONFIG["lr_decay"],
                lr_min=CONFIG["lr_min"],
                epocas=CONFIG["epocas_cv"],
                tamanho_lote=CONFIG["tamanho_lote"],
                seed=123 + 1000 * c_idx + f_idx,
                imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"],
                avaliar_loss_subamostra_por_epoca=False,
                loss_subamostra_max=0,
            )

            y_pred_va, _ = prever(modelo, X_va)
            err = calcular_erro_classificacao(y_va.astype(np.int64), y_pred_va.astype(np.int64))
            erros.append(err)
            print(f"[CV]  Fold {f_idx}/{k_folds} - erro={err:.4f}")

        mean_err = float(np.mean(erros))
        print(f"[CV] Média erro = {mean_err:.4f}")

        if (melhor is None) or (mean_err < melhor[0]):
            melhor = (mean_err, l1, l2)
            print(f"[CV] ** Novo melhor: erro={mean_err:.4f} | l1={l1} | l2={l2}")

    return melhor


# ============================================================
# Avaliação no teste: top-10 confusão + métricas
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
            frac = row.astype(np.float64) / s
            line2 = (" " * w_label) + "  " + " ".join(_fmt_float(v, w_cell, d=3) for v in frac) + " | " + _fmt_float(1.0, w_cell, d=3)
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

    print(f"[Teste] Métricas por classe (top-{k}) no teste inteiro:")
    print("classe | suporte | precision | recall | f1 | top-3 confusões (pred)")
    print("-" * 72)

    for c in top_classes.tolist():
        c = int(c)
        tp = int(np.sum((y_test == c) & (y_pred_test == c)))
        fp = int(np.sum((y_test != c) & (y_pred_test == c)))
        fn = int(np.sum((y_test == c) & (y_pred_test != c)))
        sup = int(np.sum(y_test == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        mask_c = (y_test == c)
        preds_c = y_pred_test[mask_c]
        conf_txt = "-"
        if len(preds_c) > 0:
            vals, cnts = np.unique(preds_c, return_counts=True)
            ord2 = np.argsort(cnts)[::-1]
            conf_list = []
            for idx in ord2:
                lab = int(vals[idx])
                if lab == c:
                    continue
                conf_list.append(f"{lab}:{int(cnts[idx])}")
                if len(conf_list) >= 3:
                    break
            conf_txt = ", ".join(conf_list) if conf_list else "-"

        print(f"{c:>6} | {sup:>7} | {prec:>9.3f} | {rec:>6.3f} | {f1:>4.3f} | {conf_txt}")

    print("-" * 72)


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

    print("\n[Info] Dataset original")
    print("X:", X.shape, X.dtype)
    print("y:", y.shape, y.dtype)
    print("n classes:", len(np.unique(y)))

    diagnostico_dim_hog(int(X.shape[1]))

    # Seleção de classes
    print(
        f"\n[Etapa 1/6] Selecionando {CONFIG['frac_classes']*100:.3f}% das classes ELEGÍVEIS (seed={CONFIG['seed_classes']}) "
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
        print("Sugestões: diminua min_amostras_por_classe, ou aumente frac_classes, ou ambos.")
        return

    print("[Etapa 1/6] Após filtro:")
    print("  X:", X.shape, "| n classes:", len(np.unique(y)), "| classes escolhidas:", len(escolhidas))

    # Split
    print(
        f"\n[Etapa 2/6] Split treino/teste (test_frac={CONFIG['test_frac']:.2f}) "
        f"com mínimo {CONFIG['n_min_treino_por_classe']} no treino por classe..."
    )
    X_train, X_test, y_train, y_test = split_estratificado_min_treino(
        X, y,
        test_frac=CONFIG["test_frac"],
        seed=CONFIG["seed_split"],
        min_treino_por_classe=CONFIG["n_min_treino_por_classe"]
    )

    # filtro p/ CV estratificado
    print(f"\n[Etapa 3/6] Garantindo mínimo de {CONFIG['k_folds']} amostras por classe no TREINO (para CV)...")
    X_train, y_train, X_test, y_test, classes_ok = filtrar_classes_min_train_para_cv(
        X_train, y_train, X_test, y_test, min_train_por_classe=CONFIG["k_folds"]
    )

    if X_train.shape[0] == 0 or len(np.unique(y_train)) == 0:
        print("\n[ERRO] Após garantir CV, não sobrou treino suficiente.")
        print("Sugestões: aumente frac_classes ou reduza min_amostras_por_classe.")
        return

    print("[Etapa 3/6] Shapes após filtro p/ CV:")
    print("  Train:", X_train.shape, y_train.shape, "| n classes:", len(np.unique(y_train)))
    print("  Test :", X_test.shape, y_test.shape, "| n classes:", len(np.unique(y_test)))

    classes_fixas = np.unique(y_train)

    # CV (scaler dentro do fold)
    print(f"\n[Etapa 4/6] CV estratificado (k={CONFIG['k_folds']}) para escolher lambdas...")
    best = escolher_melhores_lambdas_por_cv(
        X_train_raw=X_train,
        y_train=y_train,
        classes_fixas=classes_fixas,
        lambda_l1_grid=CONFIG["lambda_l1_grid"],
        lambda_l2_grid=CONFIG["lambda_l2_grid"],
        k_folds=CONFIG["k_folds"],
        seed_cv=CONFIG["seed_cv"],
    )
    best_mean_err, best_l1, best_l2 = best
    print(f"\n[CV] Melhor: mean_err={best_mean_err:.4f} | l1={best_l1} | l2={best_l2}")

    # scaler final (treino inteiro)
    print("\n[Etapa 5/6] Padronizando features (z-score) com stats do TREINO...")
    mean_tr, std_tr = fit_standardizer(X_train, eps=CONFIG["std_eps"])
    X_train_feat = apply_standardizer(X_train, mean_tr, std_tr)
    X_test_feat = apply_standardizer(X_test, mean_tr, std_tr)

    # treino final
    print(f"\n[Etapa 6/6] Treinando modelo final (SGD, epocas={CONFIG['epocas_final']})...")
    modelo_final = treinar_softmax_elasticnet_sgd(
        X_train_feat, y_train, classes_fixas,
        lambda_l1=best_l1, lambda_l2=best_l2,
        lr_inicial=CONFIG["lr_inicial"],
        lr_decay=CONFIG["lr_decay"],
        lr_min=CONFIG["lr_min"],
        epocas=CONFIG["epocas_final"],
        tamanho_lote=CONFIG["tamanho_lote"],
        seed=999,
        imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"],
        avaliar_loss_subamostra_por_epoca=CONFIG["avaliar_loss_subamostra_por_epoca"],
        loss_subamostra_max=CONFIG["loss_subamostra_max"],
    )

    # treino
    y_pred_train, _ = prever(modelo_final, X_train_feat)
    err_train = calcular_erro_classificacao(y_train.astype(np.int64), y_pred_train.astype(np.int64))
    print(f"\n[Resultado] TREINO: erro={err_train:.4f} | acurácia={1.0-err_train:.4f}")

    # exemplos
    if CONFIG["exemplos_de"].lower() == "treino":
        mostrar_previsoes_amostrais(y_train, y_pred_train, CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])
    else:
        y_pred_test, _ = prever(modelo_final, X_test_feat)
        mostrar_previsoes_amostrais(y_test, y_pred_test, CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])

    # teste
    print("\n[Etapa Extra] Avaliando no TESTE (acurácia + matriz de confusão top-10)...")
    y_pred_test_full, _ = prever(modelo_final, X_test_feat)
    avaliar_teste_top10(y_test=y_test, y_pred_test=y_pred_test_full, top_k=10)


if __name__ == "__main__":
    main()
