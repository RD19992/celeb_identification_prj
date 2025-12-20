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

    # prototipagem
    "frac_classes": 0.02,          # fração das CLASSES ELEGÍVEIS
    "seed_classes": 42,
    "min_amostras_por_classe": 20,  # mínimo no dataset inteiro

    # split
    "test_frac": 0.20,
    "seed_split": 42,
    "n_min_treino_por_classe": 25,  # garante esse mínimo indo pro treino

    # CV
    "k_folds": 3,
    "seed_cv": 42,

    # lambdas
    "lambda_l1_grid": [0.0, 0.05, 0.1, 0.5],
    "lambda_l2_grid": [0.0, 0.05, 0.1, 0.5],

    # treino
    "epocas_cv": 15,
    "epocas_final": 30,

    # ========================================================
    # SGD (sem Armijo): LR fixo + decaimento por época
    # lr_epoch = max(lr_min, lr_inicial * (lr_decay ** epoch))
    # ========================================================
    "lr_inicial": 0.5,
    "lr_decay": 0.95,
    "lr_min": 1e-4,

    "tamanho_lote": 256,
    "imprimir_a_cada_n_lotes": 25,

    # (X - mean_train) / (std_train + eps)
    "std_eps": 1e-8,

    # output
    "n_exemplos_previsao": 10,
    "exemplos_de": "teste",

    # diagnóstico
    "diag_seed": 42,
    "diag_max_amostras_prob": 5000,

    # Para full dataset, recomendo colocar False (fica MUITO mais rápido).
    "avaliar_loss_full_a_cada_epoca": True,
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
# Utilitários
# ============================================================

def codificar_rotulos_com_classes(y: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes)
    mapa = {int(c): i for i, c in enumerate(classes.tolist())}
    y_idx = np.array([mapa[int(v)] for v in y], dtype=np.int64)
    return y_idx, classes.astype(np.int64), mapa

def calcular_erro_classificacao(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true != y_pred))

def mostrar_previsoes_amostrais(y_true: np.ndarray, y_pred: np.ndarray, n_amostras: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(n_amostras, n), replace=False)
    print("\nExemplos aleatórios (verdadeiro -> predito):")
    for t, i in enumerate(idx, start=1):
        print(f"  #{t:02d} | y_true={int(y_true[i])} -> y_pred={int(y_pred[i])}")

def filtrar_classes_min_train_para_cv(X_train, y_train, X_test, y_test, min_train_por_classe: int):
    classes, counts = np.unique(y_train, return_counts=True)
    classes_ok = classes[counts >= min_train_por_classe]
    mask_tr = np.isin(y_train, classes_ok)
    mask_te = np.isin(y_test, classes_ok)
    return X_train[mask_tr], y_train[mask_tr], X_test[mask_te], y_test[mask_te], classes_ok

# ============================================================
# Padronização (z-score) com stats do treino
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
# Seleção de classes (CORRIGIDA): elegíveis primeiro
# ============================================================

def selecionar_classes_aleatorias_entre_elegiveis(X, y, frac_classes: float, seed: int, min_amostras_por_classe: int):
    rng = np.random.default_rng(seed)

    classes_all, counts_all = np.unique(y, return_counts=True)
    elegiveis = classes_all[counts_all >= min_amostras_por_classe]

    print(f"[Seleção] Classes elegíveis (>= {min_amostras_por_classe} amostras): {len(elegiveis)} de {len(classes_all)}")

    if len(elegiveis) == 0:
        return X[:0], y[:0], elegiveis  # vazio

    n_escolher = max(1, int(np.round(frac_classes * len(elegiveis))))
    n_escolher = min(n_escolher, len(elegiveis))

    escolhidas = rng.choice(elegiveis, size=n_escolher, replace=False)
    mask = np.isin(y, escolhidas)
    return X[mask], y[mask], escolhidas

# ============================================================
# Split garantindo treino por classe
# ============================================================

def split_garantindo_treino_por_classe(X, y, test_frac: float, seed: int, n_min_treino_por_classe: int = 1):
    rng = np.random.default_rng(seed)
    n = len(y)
    test_n = int(np.round(test_frac * n))

    order = np.argsort(y)
    y_sorted = y[order]
    _, start_idx = np.unique(y_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(y_sorted))

    train_idx = []
    resto_idx = []

    for s, e in zip(start_idx, end_idx):
        grp = order[s:e]
        rng.shuffle(grp)
        take = min(n_min_treino_por_classe, len(grp))
        train_idx.append(grp[:take])
        resto_idx.append(grp[take:])

    train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=np.int64)
    resto_idx = np.concatenate(resto_idx) if len(resto_idx) else np.array([], dtype=np.int64)

    if test_n > len(resto_idx):
        test_n = len(resto_idx)

    test_idx = rng.choice(resto_idx, size=test_n, replace=False)
    test_set = set(test_idx.tolist())
    resto_para_treino = np.array([i for i in resto_idx if i not in test_set], dtype=np.int64)

    train_idx = np.concatenate([train_idx, resto_para_treino])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ============================================================
# Softmax + Weighted CE + Elastic Net (SEM one-hot)
# ============================================================

def compute_class_weights_from_yidx(y_idx: np.ndarray, K: int, eps: float = 1e-12) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=K).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    w = total / (K * counts_safe + eps)
    return w.astype(np.float32)

def perda_e_gradiente_softmax_wce_elasticnet_yidx(
    params: dict,
    X: np.ndarray,
    y_idx: np.ndarray,
    class_weights: np.ndarray,
    lambda_l1: float,
    lambda_l2: float,
    eps: float = 1e-12
):
    """
    Weighted Cross-Entropy sem one-hot:
      loss = mean( w[y] * -log(p_y) ) + reg/m
      dlogits = w[y]*(P - onehot(y))/m

    W: (K, d)
    b: (K,)
    X: (m, d)
    y_idx: (m,)
    """
    W = params["W"]
    b = params["b"]
    m = X.shape[0]

    if m <= 0:
        return 0.0, {"W": np.zeros_like(W), "b": np.zeros_like(b)}

    X = X.astype(np.float32, copy=False)
    y_idx = y_idx.astype(np.int64, copy=False)

    # logits (m, K)
    logits = X @ W.T + b
    logits = logits - np.max(logits, axis=1, keepdims=True)

    expz = np.exp(logits).astype(np.float32, copy=False)
    P = expz / (np.sum(expz, axis=1, keepdims=True) + eps)

    # WCE
    w_per_sample = class_weights[y_idx].astype(np.float32, copy=False)
    p_true = P[np.arange(m), y_idx]
    wce = -float(np.mean(w_per_sample.astype(np.float64) * np.log(p_true + eps)))

    # reg (/m conforme sua versão atual)
    reg = (lambda_l1 * float(np.sum(np.abs(W))) + 0.5 * lambda_l2 * float(np.sum(W * W))) / float(m)
    loss = wce + reg

    # grad (truque padrão)
    dlogits = P  # in-place ok (não usamos P depois disso)
    dlogits[np.arange(m), y_idx] -= 1.0
    dlogits *= (w_per_sample[:, None] / float(m)).astype(np.float32, copy=False)

    grad_W = dlogits.T @ X                      # (K, d)
    grad_b = np.sum(dlogits, axis=0)            # (K,)

    grad_W += (lambda_l2 * W) / float(m)
    grad_W += (lambda_l1 * np.sign(W)) / float(m)

    return float(loss), {"W": grad_W.astype(np.float32, copy=False), "b": grad_b.astype(np.float32, copy=False)}

# ============================================================
# SGD/MBGD com LR fixo + decaimento
# ============================================================

def gradiente_descendente_lr_decay(
    params_iniciais: dict,
    funcao_perda_grad,
    X: np.ndarray,
    y_idx: np.ndarray,
    lr_inicial: float,
    lr_decay: float,
    lr_min: float,
    epocas: int,
    tamanho_lote: int | None,
    seed: int,
    verbose: bool,
    imprimir_a_cada_n_lotes: int = 25,
    avaliar_loss_full_a_cada_epoca: bool = True
):
    rng = np.random.default_rng(seed)

    params = {k: np.array(v, copy=True) for k, v in params_iniciais.items()}

    n = X.shape[0]
    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n:
        tamanho_lote = n

    n_lotes = int(np.ceil(n / tamanho_lote))
    historico = []
    stats = {
        "updates_total": 0,
        "lr_inicial": float(lr_inicial),
        "lr_decay": float(lr_decay),
        "lr_min": float(lr_min),
        "lr_min_observado": float("inf"),
        "lr_max_observado": 0.0,
    }

    for epoca in range(epocas):
        lr_epoca = max(lr_min, lr_inicial * (lr_decay ** epoca))
        stats["lr_min_observado"] = min(stats["lr_min_observado"], lr_epoca)
        stats["lr_max_observado"] = max(stats["lr_max_observado"], lr_epoca)

        idx = rng.permutation(n)

        for num_lote, ini in enumerate(range(0, n, tamanho_lote), start=1):
            lote = idx[ini:ini + tamanho_lote]
            Xb = X[lote]
            yb = y_idx[lote]

            loss_b, grads = funcao_perda_grad(params, Xb, yb)

            params["W"] -= lr_epoca * grads["W"]
            params["b"] -= lr_epoca * grads["b"]
            stats["updates_total"] += 1

            if (num_lote % imprimir_a_cada_n_lotes == 0) or (num_lote == n_lotes):
                print(
                    f"[Treino] Época {epoca+1}/{epocas} | Lote {num_lote}/{n_lotes} | "
                    f"lr={lr_epoca:.3e} | loss_batch={loss_b:.4f}",
                    end="\r"
                )

        print(" " * 120, end="\r")

        if verbose and avaliar_loss_full_a_cada_epoca:
            loss_full, _ = funcao_perda_grad(params, X, y_idx)
            historico.append(float(loss_full))
            print(f"[SGD] Época {epoca+1}/{epocas} | lr={lr_epoca:.3e} | loss_full={loss_full:.6f}")
        elif verbose:
            print(f"[SGD] Época {epoca+1}/{epocas} | lr={lr_epoca:.3e}")

    return params, historico, stats

# ============================================================
# Treino / Predição
# ============================================================

def treinar_regressao_linear_multiclasse_elasticnet(
    X_treino, y_treino, classes_fixas,
    lambda_l1, lambda_l2,
    lr_inicial, lr_decay, lr_min,
    epocas, tamanho_lote,
    seed, verbose, imprimir_a_cada_n_lotes,
    avaliar_loss_full_a_cada_epoca: bool = True
):
    y_idx, classes, _ = codificar_rotulos_com_classes(y_treino, classes_fixas)
    K = len(classes)
    d = X_treino.shape[1]

    class_weights = compute_class_weights_from_yidx(y_idx, K)

    rng = np.random.default_rng(seed)
    W0 = rng.normal(0.0, 0.01, size=(K, d)).astype(np.float32)
    b0 = np.zeros(K, dtype=np.float32)
    params0 = {"W": W0, "b": b0}

    def f(params, Xb, yb_idx):
        return perda_e_gradiente_softmax_wce_elasticnet_yidx(
            params, Xb, yb_idx,
            class_weights=class_weights,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2
        )

    params_finais, historico, stats_gd = gradiente_descendente_lr_decay(
        params0, f,
        X_treino.astype(np.float32, copy=False),
        y_idx,
        lr_inicial=lr_inicial,
        lr_decay=lr_decay,
        lr_min=lr_min,
        epocas=epocas,
        tamanho_lote=tamanho_lote,
        seed=seed,
        verbose=verbose,
        imprimir_a_cada_n_lotes=imprimir_a_cada_n_lotes,
        avaliar_loss_full_a_cada_epoca=avaliar_loss_full_a_cada_epoca
    )

    return {
        "W": params_finais["W"],
        "b": params_finais["b"],
        "classes": classes,
        "historico_perda": historico,
        "lambda_l1": float(lambda_l1),
        "lambda_l2": float(lambda_l2),
        "class_weights": class_weights,
        "gd_stats": stats_gd,
    }

def prever_regressao_linear_multiclasse(modelo, X):
    W, b, classes = modelo["W"], modelo["b"], modelo["classes"]
    logits = X.astype(np.float32, copy=False) @ W.T + b
    idx_pred = np.argmax(logits, axis=1)
    return classes[idx_pred], logits

# ============================================================
# Diagnóstico (pós-treino)
# ============================================================

def diagnostico_modelo(modelo, y_train, y_pred_train, logits_train=None, titulo="Diagnóstico", max_amostras_prob=5000, seed=42):
    rng = np.random.default_rng(seed)
    classes = modelo["classes"]
    K = len(classes)
    n = len(y_train)

    print("\n" + "=" * 72)
    print(f"[{titulo}]")
    print("=" * 72)
    print(f"N treino = {n} | K (classes) = {K}")
    print(f"Baseline aleatório 1/K = {1.0 / max(K, 1):.6f}")

    vals, cnt = np.unique(y_train, return_counts=True)
    maj = float(np.max(cnt) / n) if n > 0 else 0.0
    print(f"Baseline maioria = {maj:.6f} | y_train min/med/max = {int(np.min(cnt))}/{int(np.median(cnt))}/{int(np.max(cnt))}")

    vals_p, cnt_p = np.unique(y_pred_train, return_counts=True)
    cobertura = len(vals_p) / max(K, 1)
    top = np.argsort(cnt_p)[::-1][:10]
    print(f"Predições únicas = {len(vals_p)}/{K} (cobertura={cobertura:.3f})")
    print("Top-10 classes preditas (classe, count, fração):")
    for i in top:
        print(f"  {int(vals_p[i])} | {int(cnt_p[i])} | {cnt_p[i] / n:.4f}")

    W = modelo["W"]
    absW = np.abs(W)
    print(f"||W||_2={float(np.linalg.norm(W)):.3e} | max|W|={float(np.max(absW)):.3e} | sparsity(|W|<1e-10)={float(np.mean(absW<1e-10)):.3f}")

    if logits_train is not None and n > 0:
        m = min(n, max_amostras_prob)
        idx = rng.choice(n, size=m, replace=False)
        L = logits_train[idx].astype(np.float32, copy=False)
        z = L - np.max(L, axis=1, keepdims=True)
        expz = np.exp(z).astype(np.float32, copy=False)
        P = expz / (np.sum(expz, axis=1, keepdims=True) + 1e-12)
        pmax = np.max(P, axis=1)
        print(f"Confiança (amostra={m}): mean(max softmax)={float(np.mean(pmax)):.4f} | med={float(np.median(pmax)):.4f} | p90={float(np.quantile(pmax, 0.90)):.4f}")

    st = modelo.get("gd_stats", {})
    if st:
        upd = max(int(st.get("updates_total", 0)), 1)
        print(
            f"SGD: updates={upd} | lr0={st.get('lr_inicial'):.3e} | decay={st.get('lr_decay'):.4f} | "
            f"lr_min={st.get('lr_min'):.3e} | lr_min_obs={st.get('lr_min_observado'):.3e} | lr_max_obs={st.get('lr_max_observado'):.3e}"
        )

    print("=" * 72 + "\n")

# ============================================================
# CV
# ============================================================

def escolher_melhores_lambdas_por_cv(X_train, y_train, classes_fixas, lambda_l1_grid, lambda_l2_grid, k_folds, seed_cv, treino_cfg):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_cv)
    combos = list(product(lambda_l1_grid, lambda_l2_grid))
    melhor = None

    total_combos = len(combos)
    print(f"[CV] Grid-search: {total_combos} combinações | {k_folds}-fold estratificado")

    for c_idx, (l1, l2) in enumerate(combos, start=1):
        erros = []
        print(f"\n[CV] Combo {c_idx}/{total_combos}: l1={l1} | l2={l2}")

        for f_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            print(f"[CV]  Fold {f_idx}/{k_folds} - treinando...")

            modelo = treinar_regressao_linear_multiclasse_elasticnet(
                X_tr, y_tr, classes_fixas,
                lambda_l1=l1, lambda_l2=l2,
                lr_inicial=treino_cfg["lr_inicial"],
                lr_decay=treino_cfg["lr_decay"],
                lr_min=treino_cfg["lr_min"],
                epocas=treino_cfg["epocas_cv"],
                tamanho_lote=treino_cfg["tamanho_lote"],
                seed=treino_cfg["seed_treino"] + f_idx,
                verbose=False,
                imprimir_a_cada_n_lotes=treino_cfg["imprimir_a_cada_n_lotes"],
                avaliar_loss_full_a_cada_epoca=treino_cfg.get("avaliar_loss_full_a_cada_epoca", False),
            )

            y_pred_va, _ = prever_regressao_linear_multiclasse(modelo, X_va)
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
# Avaliação de teste: matriz de confusão top-10
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

def avaliar_teste_top10(y_test: np.ndarray, y_pred_test: np.ndarray, top_k: int = 10, seed: int = 42):
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
        else:
            conf_txt = "-"

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

    # ---- seleção corrigida
    print(
        f"\n[Etapa 1/5] Selecionando {CONFIG['frac_classes']*100:.3f}% das classes ELEGÍVEIS (seed={CONFIG['seed_classes']}) "
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

    print("[Etapa 1/5] Após filtro:")
    print("  X:", X.shape, "| n classes:", len(np.unique(y)), "| classes escolhidas:", len(escolhidas))

    # split
    print(
        f"\n[Etapa 2/5] Split treino/teste (test_frac={CONFIG['test_frac']:.2f}) "
        f"garantindo {CONFIG['n_min_treino_por_classe']}+ no treino..."
    )
    X_train, X_test, y_train, y_test = split_garantindo_treino_por_classe(
        X, y,
        test_frac=CONFIG["test_frac"],
        seed=CONFIG["seed_split"],
        n_min_treino_por_classe=CONFIG["n_min_treino_por_classe"]
    )

    # garante CV viável (>= k_folds por classe no treino)
    print(f"\n[Etapa 3/5] Garantindo mínimo de {CONFIG['k_folds']} amostras por classe no TREINO (para CV)...")
    X_train, y_train, X_test, y_test, _ = filtrar_classes_min_train_para_cv(
        X_train, y_train, X_test, y_test, min_train_por_classe=CONFIG["k_folds"]
    )

    if X_train.shape[0] == 0 or len(np.unique(y_train)) == 0:
        print("\n[ERRO] Após garantir CV, não sobrou treino suficiente.")
        print("Sugestões: aumente frac_classes ou reduza min_amostras_por_classe.")
        return

    print("[Etapa 3/5] Shapes após filtro p/ CV:")
    print("  Train:", X_train.shape, y_train.shape, "| n classes:", len(np.unique(y_train)))
    print("  Test :", X_test.shape, y_test.shape, "| n classes:", len(np.unique(y_test)))

    # ---- z-score
    print("\n[Etapa 3.5/5] Padronizando: (X - mean_train) / (std_train + eps)")
    mean_train, std_train = fit_standardizer(X_train, eps=CONFIG["std_eps"])
    X_train_feat = apply_standardizer(X_train, mean_train, std_train)
    X_test_feat = apply_standardizer(X_test, mean_train, std_train)

    classes_fixas = np.unique(y_train)

    # CV
    print(f"\n[Etapa 4/5] CV estratificado (k={CONFIG['k_folds']}) para escolher lambdas...")
    treino_cfg = {
        "lr_inicial": CONFIG["lr_inicial"],
        "lr_decay": CONFIG["lr_decay"],
        "lr_min": CONFIG["lr_min"],
        "epocas_cv": CONFIG["epocas_cv"],
        "tamanho_lote": CONFIG["tamanho_lote"],
        "seed_treino": 123,
        "imprimir_a_cada_n_lotes": CONFIG["imprimir_a_cada_n_lotes"],
        "avaliar_loss_full_a_cada_epoca": False,  # CV mais rápido
    }

    best = escolher_melhores_lambdas_por_cv(
        X_train_feat, y_train, classes_fixas,
        CONFIG["lambda_l1_grid"], CONFIG["lambda_l2_grid"],
        k_folds=CONFIG["k_folds"],
        seed_cv=CONFIG["seed_cv"],
        treino_cfg=treino_cfg
    )
    best_mean_err, best_l1, best_l2 = best
    print(f"\n[CV] Melhor: mean_err={best_mean_err:.4f} | l1={best_l1} | l2={best_l2}")

    # treino final
    print(f"\n[Etapa 5/5] Treinando modelo final (epocas={CONFIG['epocas_final']})...")
    modelo_final = treinar_regressao_linear_multiclasse_elasticnet(
        X_train_feat, y_train, classes_fixas,
        lambda_l1=best_l1,
        lambda_l2=best_l2,
        lr_inicial=CONFIG["lr_inicial"],
        lr_decay=CONFIG["lr_decay"],
        lr_min=CONFIG["lr_min"],
        epocas=CONFIG["epocas_final"],
        tamanho_lote=CONFIG["tamanho_lote"],
        seed=999,
        verbose=True,
        imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"],
        avaliar_loss_full_a_cada_epoca=CONFIG["avaliar_loss_full_a_cada_epoca"],
    )

    # erro treino
    y_pred_train, logits_train = prever_regressao_linear_multiclasse(modelo_final, X_train_feat)
    erro_treino = calcular_erro_classificacao(y_train.astype(np.int64), y_pred_train.astype(np.int64))
    print(f"\n[Resultado] Erro no TREINO: {erro_treino:.4f} | Acurácia: {1.0-erro_treino:.4f}")

    diagnostico_modelo(
        modelo_final,
        y_train=y_train.astype(np.int64),
        y_pred_train=y_pred_train.astype(np.int64),
        logits_train=logits_train,
        titulo="Diagnóstico PÓS-TREINO (modelo final)",
        max_amostras_prob=CONFIG["diag_max_amostras_prob"],
        seed=CONFIG["diag_seed"]
    )

    # exemplos
    if CONFIG["exemplos_de"].lower() == "treino":
        y_true_ex, y_pred_ex = y_train, y_pred_train
    else:
        y_pred_test, _ = prever_regressao_linear_multiclasse(modelo_final, X_test_feat)
        y_true_ex, y_pred_ex = y_test, y_pred_test

    mostrar_previsoes_amostrais(
        y_true_ex.astype(np.int64),
        y_pred_ex.astype(np.int64),
        n_amostras=CONFIG["n_exemplos_previsao"],
        seed=CONFIG["seed_split"]
    )

    # avaliação teste
    print("\n[Etapa Extra] Avaliando no TESTE (acurácia + matriz de confusão top-10)...")
    y_pred_test_full, _ = prever_regressao_linear_multiclasse(modelo_final, X_test_feat)
    avaliar_teste_top10(
        y_test=y_test,
        y_pred_test=y_pred_test_full,
        top_k=10,
        seed=CONFIG.get("seed_split", 42)
    )

if __name__ == "__main__":
    main()
