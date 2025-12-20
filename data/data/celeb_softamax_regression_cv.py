import joblib
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import StratifiedKFold

# ============================================================
# CONFIG (ajuste aqui para prototipagem rápida)
# ============================================================

CONFIG = {
    # Dataset
    "dataset_filename": "celeba_hog_128x128_o9.joblib",

    # Seleção de classes ANTES do split (prototipagem)
    "frac_classes": 0.005,         # <- pedido: 0.005
    "seed_classes": 42,

    # mínimo de amostras por classe no filtro inicial
    "min_amostras_por_classe": 50,  # <- pedido: 50

    # Split treino/teste (sem validação)
    "test_frac": 0.09,
    "seed_split": 42,

    # garante pelo menos N amostras por classe no treino
    "n_min_treino_por_classe": 5,

    # CV
    "k_folds": 3,
    "seed_cv": 42,

    # Grid de lambdas (Elastic Net)
    "lambda_l1_grid": [0.0, 0.05, 0.1, 0.5],
    "lambda_l2_grid": [0.0, 0.05, 0.1, 0.5],

    # Treino por GD com Armijo (taxa_aprendizado = alpha inicial do Armijo)
    # <- pedido: aumentar alpha inicial
    "epocas_cv": 15,
    "epocas_final": 30,
    "taxa_aprendizado": 10.0,   # <- AQUI: alpha inicial grande (Armijo faz backtracking se preciso)
    "tamanho_lote": 256,

    # Progresso do GD
    "imprimir_a_cada_n_lotes": 25,

    # Padronização (z-score)
    "std_eps": 1e-8,

    # Output
    "n_exemplos_previsao": 10,
    "exemplos_de": "teste",  # "teste" ou "treino"

    # Diagnóstico
    "diag_seed": 42,
    "diag_max_amostras_prob": 5000,
}

# ============================================================
# Utilitários
# ============================================================

def diagnostico_dim_hog(d: int):
    """
    Diagnóstico heurístico do tamanho do HOG.
    Com parâmetros típicos (o=9, cell=8x8, block=2x2):
      - 64x64 -> 1764
      - 128x128 -> 8100
    Também estima IMG_SIZE provável caso d seja diferente.
    """
    print("\n[Diagnóstico HOG] Dimensão de features (d) =", d)

    if d == 1764:
        print("[Diagnóstico HOG] d=1764 sugere HOG de 64x64 (o9, cell8, block2).")
        return
    if d == 8100:
        print("[Diagnóstico HOG] d=8100 sugere HOG de 128x128 (o9, cell8, block2).")
        return

    # Estimativa aproximada assumindo:
    # dims = (cells-1)^2 * (2*2*9) = 36*(cells-1)^2, com cells = IMG/8
    # d ≈ 36*(IMG/8 - 1)^2  => IMG ≈ 8*(sqrt(d/36)+1)
    est_cells_minus_1 = np.sqrt(max(d, 1) / 36.0)
    est_img = 8.0 * (est_cells_minus_1 + 1.0)
    print("[Diagnóstico HOG] d não bate com 64x64/128x128 típicos.")
    print(f"[Diagnóstico HOG] Estimativa grosseira de IMG_SIZE ≈ {est_img:.1f}px (assumindo o9, cell8, block2).")
    print("[Diagnóstico HOG] Se isso estiver errado, confirme parâmetros do HOG e se o joblib carregado é o esperado.")

def codificar_rotulos_com_classes(y: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes)
    mapa = {int(c): i for i, c in enumerate(classes.tolist())}
    y_idx = np.array([mapa[int(v)] for v in y], dtype=np.int64)
    return y_idx, classes.astype(np.int64), mapa

def one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    n = y_idx.shape[0]
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y_idx] = 1.0
    return Y

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
# Padronização: (X - mean_train) / (std_train + eps)
# ============================================================

def fit_standardizer(X_train: np.ndarray, eps: float):
    X_train = X_train.astype(np.float64, copy=False)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.maximum(std, eps)
    return mean, std

def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X = X.astype(np.float64, copy=False)
    return (X - mean) / std

# ============================================================
# Seleção de classes ANTES do split
# ============================================================

def selecionar_classes_aleatorias(X, y, frac_classes: float, seed: int, min_amostras_por_classe: int):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_classes_total = len(classes)

    n_escolher = max(1, int(np.round(frac_classes * n_classes_total)))
    escolhidas = rng.choice(classes, size=n_escolher, replace=False)

    mask = np.isin(y, escolhidas)
    X2, y2 = X[mask], y[mask]

    c2, counts = np.unique(y2, return_counts=True)
    classes_ok = c2[counts >= min_amostras_por_classe]
    mask2 = np.isin(y2, classes_ok)

    X3, y3 = X2[mask2], y2[mask2]
    return X3, y3, classes_ok

# ============================================================
# Split garantindo pelo menos n_min_treino_por_classe no treino
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
# GD com Armijo (backtracking) + progresso + stats
# ============================================================

def gradiente_descendente(
    params_iniciais: dict,
    funcao_perda_grad,
    X: np.ndarray,
    Y: np.ndarray,
    taxa_aprendizado: float,   # alpha inicial (Armijo parte daqui e reduz se necessário)
    epocas: int,
    tamanho_lote: int | None,
    seed: int,
    verbose: bool,
    imprimir_a_cada_n_lotes: int = 25
):
    rng = np.random.default_rng(seed)

    c = 1e-4
    rho = 0.5
    max_backtracks = 12
    alpha_min = 1e-10

    params = {}
    for k, v in params_iniciais.items():
        params[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v

    n = X.shape[0]
    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n:
        tamanho_lote = n

    n_lotes = int(np.ceil(n / tamanho_lote))
    historico = []

    stats = {
        "updates_total": 0,
        "armijo_aceitou": 0,
        "armijo_fallback": 0,
        "alpha_sum": 0.0,
        "alpha_min": alpha_min,
        "alpha_max_observado": 0.0,
        "backtracks_sum": 0,
    }

    for epoca in range(epocas):
        idx = rng.permutation(n)

        for num_lote, ini in enumerate(range(0, n, tamanho_lote), start=1):
            lote = idx[ini:ini + tamanho_lote]
            Xb = X[lote]
            Yb = Y[lote]

            perda0, grads = funcao_perda_grad(params, Xb, Yb)

            grad_norm_sq = 0.0
            for g in grads.values():
                grad_norm_sq += float(np.sum(g * g))
            if grad_norm_sq <= 0.0:
                continue

            alpha = float(taxa_aprendizado)  # <- agora começa grande (por CONFIG)
            bt = max_backtracks

            aceitou = False
            while True:
                params_cand = {}
                for nome_param in params:
                    if nome_param in grads:
                        params_cand[nome_param] = params[nome_param] - alpha * grads[nome_param]
                    else:
                        params_cand[nome_param] = params[nome_param]

                perda_cand, _ = funcao_perda_grad(params_cand, Xb, Yb)

                if perda_cand <= perda0 - c * alpha * grad_norm_sq:
                    aceitou = True
                    params = params_cand
                    break

                alpha *= rho
                bt -= 1
                if alpha < alpha_min or bt <= 0:
                    break

            stats["updates_total"] += 1
            stats["alpha_sum"] += float(alpha)
            stats["alpha_max_observado"] = max(stats["alpha_max_observado"], float(alpha))
            stats["backtracks_sum"] += int(max_backtracks - bt)

            if aceitou:
                stats["armijo_aceitou"] += 1
            else:
                stats["armijo_fallback"] += 1
                alpha = max(alpha, alpha_min)
                for nome_param in grads:
                    params[nome_param] = params[nome_param] - alpha * grads[nome_param]

            if (num_lote % imprimir_a_cada_n_lotes == 0) or (num_lote == n_lotes):
                print(f"[Treino] Época {epoca+1}/{epocas} | Lote {num_lote}/{n_lotes}", end="\r")

        print(" " * 90, end="\r")
        if verbose:
            perda_full, _ = funcao_perda_grad(params, X, Y)
            historico.append(float(perda_full))
            print(f"[GD+Armijo] Época {epoca+1}/{epocas} | perda={perda_full:.6f}")

    return params, historico, stats

# ============================================================
# Softmax + Weighted Cross-Entropy + Elastic Net (/m correto)
# ============================================================

def compute_class_weights_from_yidx(y_idx: np.ndarray, K: int, eps: float = 1e-12) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=K).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    total = float(np.sum(counts))
    w = total / (K * counts_safe + eps)
    return w.astype(np.float64)

def perda_e_gradiente_softmax_wce_elasticnet(
    params,
    X,
    Y,
    class_weights,   # shape (K,)
    lambda_l1,
    lambda_l2,
    eps: float = 1e-12
):
    W = params["W"]  # (K, d)
    b = params["b"]  # (K,)
    m = X.shape[0]
    if m <= 0:
        return 0.0, {"W": np.zeros_like(W), "b": np.zeros_like(b)}

    logits = X @ W.T + b  # (m, K)

    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    sumexp = np.sum(expz, axis=1, keepdims=True)
    P = expz / (sumexp + eps)

    w_per_sample = (Y * class_weights[None, :]).sum(axis=1)  # (m,)

    logp_true = np.sum(Y * np.log(P + eps), axis=1)           # (m,)
    wce = -np.mean(w_per_sample * logp_true)

    reg = (lambda_l1 * np.sum(np.abs(W)) + 0.5 * lambda_l2 * np.sum(W * W)) / m
    loss = wce + reg

    dlogits = (w_per_sample[:, None] * (P - Y)) / m          # (m, K)

    grad_W = dlogits.T @ X
    grad_b = np.sum(dlogits, axis=0)

    grad_W += (lambda_l2 * W) / m
    grad_W += (lambda_l1 * np.sign(W)) / m

    return float(loss), {"W": grad_W, "b": grad_b}

# ============================================================
# Treino / Predição
# ============================================================

def treinar_regressao_linear_multiclasse_elasticnet(
    X_treino, y_treino, classes_fixas,
    lambda_l1, lambda_l2,
    taxa_aprendizado, epocas, tamanho_lote,
    seed, verbose, imprimir_a_cada_n_lotes
):
    y_idx, classes, _ = codificar_rotulos_com_classes(y_treino, classes_fixas)
    K = len(classes)
    d = X_treino.shape[1]
    Y = one_hot(y_idx, K)

    class_weights = compute_class_weights_from_yidx(y_idx, K)

    rng = np.random.default_rng(seed)
    W0 = rng.normal(0.0, 0.01, size=(K, d)).astype(np.float64)
    b0 = np.zeros(K, dtype=np.float64)
    params0 = {"W": W0, "b": b0}

    def f(params, Xb, Yb):
        Xb = Xb.astype(np.float64, copy=False)
        return perda_e_gradiente_softmax_wce_elasticnet(
            params, Xb, Yb,
            class_weights=class_weights,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2
        )

    params_finais, historico, stats_gd = gradiente_descendente(
        params0, f,
        X_treino.astype(np.float64, copy=False),
        Y,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        tamanho_lote=tamanho_lote,
        seed=seed,
        verbose=verbose,
        imprimir_a_cada_n_lotes=imprimir_a_cada_n_lotes
    )

    return {
        "W": params_finais["W"],
        "b": params_finais["b"],
        "classes": classes,
        "historico_perda": historico,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "class_weights": class_weights,
        "gd_stats": stats_gd,
    }

def prever_regressao_linear_multiclasse(modelo, X):
    W, b, classes = modelo["W"], modelo["b"], modelo["classes"]
    logits = X.astype(np.float64, copy=False) @ W.T + b
    idx_pred = np.argmax(logits, axis=1)
    return classes[idx_pred], logits

# ============================================================
# Diagnóstico
# ============================================================

def diagnostico_modelo(
    *,
    modelo,
    y_train,
    y_pred_train,
    logits_train=None,
    titulo: str = "Diagnóstico",
    max_amostras_prob: int = 5000,
    seed: int = 42
):
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
    print(f"Baseline maioria (classe mais frequente) = {maj:.6f}")
    print(f"Distribuição y_train: min={int(np.min(cnt))} | med={int(np.median(cnt))} | max={int(np.max(cnt))}")

    vals_p, cnt_p = np.unique(y_pred_train, return_counts=True)
    cobertura = len(vals_p) / max(K, 1)
    top = np.argsort(cnt_p)[::-1][:10]
    print(f"Predições únicas = {len(vals_p)}/{K}  (cobertura={cobertura:.3f})")
    print("Top-10 classes preditas (classe, count, fração):")
    for i in top:
        print(f"  {int(vals_p[i])} | {int(cnt_p[i])} | {cnt_p[i] / n:.4f}")

    W = modelo["W"]
    absW = np.abs(W)
    w_l2 = float(np.linalg.norm(W))
    w_l1 = float(np.sum(absW))
    w_max = float(np.max(absW))
    spars = float(np.mean(absW < 1e-10))
    print(f"||W||_2 = {w_l2:.6e} | ||W||_1 = {w_l1:.6e} | max|W| = {w_max:.6e} | sparsity(|W|<1e-10)={spars:.3f}")

    b = modelo["b"]
    print(f"b: mean={float(np.mean(b)):.3e} | std={float(np.std(b)):.3e} | min={float(np.min(b)):.3e} | max={float(np.max(b)):.3e}")

    cw = modelo.get("class_weights")
    if cw is not None:
        print(f"class_weights: min={float(np.min(cw)):.4f} | med={float(np.median(cw)):.4f} | max={float(np.max(cw)):.4f}")

    if logits_train is not None and n > 0:
        m = min(n, max_amostras_prob)
        idx = rng.choice(n, size=m, replace=False)

        L = logits_train[idx]
        z = L - np.max(L, axis=1, keepdims=True)
        expz = np.exp(z)
        P = expz / (np.sum(expz, axis=1, keepdims=True) + 1e-12)
        pmax = np.max(P, axis=1)

        print(f"Confiança (amostra={m}): mean(max softmax)={float(np.mean(pmax)):.4f} | med={float(np.median(pmax)):.4f} | p90={float(np.quantile(pmax, 0.90)):.4f}")

    st = modelo.get("gd_stats", {})
    if st:
        upd = max(int(st.get("updates_total", 0)), 1)
        acc = int(st.get("armijo_aceitou", 0))
        fb = int(st.get("armijo_fallback", 0))
        alpha_mean = float(st.get("alpha_sum", 0.0)) / upd
        bt_mean = float(st.get("backtracks_sum", 0)) / upd
        print(f"Armijo: aceitou={acc}/{upd} ({acc/upd:.2%}) | fallback={fb}/{upd} ({fb/upd:.2%}) | alpha_médio={alpha_mean:.3e} | backtracks_médio={bt_mean:.2f}")

    print("=" * 72 + "\n")

# ============================================================
# K-Fold CV para escolher lambdas
# ============================================================

def escolher_melhores_lambdas_por_cv(
    X_train, y_train, classes_fixas,
    lambda_l1_grid, lambda_l2_grid,
    k_folds, seed_cv,
    treino_cfg
):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed_cv)
    combos = list(product(lambda_l1_grid, lambda_l2_grid))
    melhor = None  # (mean_err, l1, l2)

    total_combos = len(combos)
    print(f"[CV] Iniciando grid-search: {total_combos} combinações | {k_folds}-fold (estratificado)")

    for c_idx, (l1, l2) in enumerate(combos, start=1):
        erros = []
        print(f"\n[CV] Combo {c_idx}/{total_combos}: lambda_l1={l1} | lambda_l2={l2}")

        for f_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            print(f"[CV]  Fold {f_idx}/{k_folds} - treinando...")

            modelo = treinar_regressao_linear_multiclasse_elasticnet(
                X_tr, y_tr, classes_fixas,
                lambda_l1=l1, lambda_l2=l2,
                taxa_aprendizado=treino_cfg["taxa_aprendizado"],
                epocas=treino_cfg["epocas_cv"],
                tamanho_lote=treino_cfg["tamanho_lote"],
                seed=treino_cfg["seed_treino"] + f_idx,
                verbose=False,
                imprimir_a_cada_n_lotes=treino_cfg["imprimir_a_cada_n_lotes"],
            )

            y_pred_va, _ = prever_regressao_linear_multiclasse(modelo, X_va)
            err = calcular_erro_classificacao(y_va.astype(np.int64), y_pred_va.astype(np.int64))
            erros.append(err)
            print(f"[CV]  Fold {f_idx}/{k_folds} - erro={err:.4f}")

        mean_err = float(np.mean(erros))
        print(f"[CV] Média erro (combo {c_idx}/{total_combos}) = {mean_err:.4f}")

        if (melhor is None) or (mean_err < melhor[0]):
            melhor = (mean_err, l1, l2)
            print(f"[CV] ** Novo melhor até agora: erro={mean_err:.4f} | l1={l1} | l2={l2}")

    return melhor

# ============================================================
# PIPELINE PRINCIPAL (SEM L2-NORM, COM Z-SCORE)
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

    # Diagnóstico inicial do tamanho do HOG (d)
    diagnostico_dim_hog(int(X.shape[1]))

    # Seleção de classes ANTES do split
    print(
        f"\n[Etapa 1/5] Selecionando {CONFIG['frac_classes']*100:.3f}% das classes (seed={CONFIG['seed_classes']}) "
        f"com mínimo {CONFIG['min_amostras_por_classe']} amostras/classe..."
    )
    X, y, _ = selecionar_classes_aleatorias(
        X, y,
        frac_classes=CONFIG["frac_classes"],
        seed=CONFIG["seed_classes"],
        min_amostras_por_classe=CONFIG["min_amostras_por_classe"]
    )
    print("[Etapa 1/5] Após filtro:")
    print("  X:", X.shape, " | n classes:", len(np.unique(y)))

    # Split treino/teste
    print(
        f"\n[Etapa 2/5] Split treino/teste (test_frac={CONFIG['test_frac']:.2f}) "
        f"garantindo {CONFIG['n_min_treino_por_classe']}+ amostras/classe no treino..."
    )
    X_train, X_test, y_train, y_test = split_garantindo_treino_por_classe(
        X, y,
        test_frac=CONFIG["test_frac"],
        seed=CONFIG["seed_split"],
        n_min_treino_por_classe=CONFIG["n_min_treino_por_classe"]
    )

    # Garante CV viável (cada classe precisa >= k_folds no treino)
    print(f"\n[Etapa 3/5] Garantindo mínimo de {CONFIG['k_folds']} amostras por classe no TREINO (para CV estratificado)...")
    X_train, y_train, X_test, y_test, _ = filtrar_classes_min_train_para_cv(
        X_train, y_train, X_test, y_test, min_train_por_classe=CONFIG["k_folds"]
    )

    print("[Etapa 3/5] Shapes após filtro p/ CV:")
    print("  Train:", X_train.shape, y_train.shape, "| n classes:", len(np.unique(y_train)))
    print("  Test :", X_test.shape, y_test.shape,  "| n classes:", len(np.unique(y_test)))

    # ===============================
    # Padronização (z-score) SOMENTE
    # ===============================
    print("\n[Etapa 3.5/5] Padronizando features: (X - mean_train) / (std_train + eps) ...")
    mean_train, std_train = fit_standardizer(X_train, eps=CONFIG["std_eps"])
    X_train_feat = apply_standardizer(X_train, mean_train, std_train)
    X_test_feat = apply_standardizer(X_test, mean_train, std_train)

    # Classes fixas
    classes_fixas = np.unique(y_train)

    # Diagnóstico pré-CV (somente baselines)
    y_pred_dummy = np.full_like(y_train, fill_value=classes_fixas[0])
    diagnostico_modelo(
        modelo={"W": np.zeros((len(classes_fixas), X_train_feat.shape[1])), "b": np.zeros(len(classes_fixas)), "classes": classes_fixas, "gd_stats": {}},
        y_train=y_train.astype(np.int64),
        y_pred_train=y_pred_dummy.astype(np.int64),
        logits_train=None,
        titulo="Diagnóstico PRÉ-CV (apenas dataset/baselines)",
        max_amostras_prob=CONFIG["diag_max_amostras_prob"],
        seed=CONFIG["diag_seed"]
    )

    # CV para escolher lambdas
    print(f"\n[Etapa 4/5] K-Fold CV (k={CONFIG['k_folds']}) para escolher lambdas...")
    treino_cfg = {
        "taxa_aprendizado": CONFIG["taxa_aprendizado"],  # <- alpha inicial alto
        "epocas_cv": CONFIG["epocas_cv"],
        "tamanho_lote": CONFIG["tamanho_lote"],
        "seed_treino": 123,
        "imprimir_a_cada_n_lotes": CONFIG["imprimir_a_cada_n_lotes"]
    }

    best = escolher_melhores_lambdas_por_cv(
        X_train_feat, y_train, classes_fixas,
        CONFIG["lambda_l1_grid"], CONFIG["lambda_l2_grid"],
        k_folds=CONFIG["k_folds"],
        seed_cv=CONFIG["seed_cv"],
        treino_cfg=treino_cfg
    )
    best_mean_err, best_l1, best_l2 = best
    print(f"\n[CV] Melhor combinação final: mean_err={best_mean_err:.4f} | lambda_l1={best_l1} | lambda_l2={best_l2}")

    # Treina modelo final no treino inteiro
    print(f"\n[Etapa 5/5] Treinando modelo final no treino inteiro (epocas={CONFIG['epocas_final']})...")
    modelo_final = treinar_regressao_linear_multiclasse_elasticnet(
        X_train_feat, y_train, classes_fixas,
        lambda_l1=best_l1,
        lambda_l2=best_l2,
        taxa_aprendizado=CONFIG["taxa_aprendizado"],
        epocas=CONFIG["epocas_final"],
        tamanho_lote=CONFIG["tamanho_lote"],
        seed=999,
        verbose=True,
        imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"],
    )

    # Erro no treino do melhor modelo
    y_pred_train, logits_train = prever_regressao_linear_multiclasse(modelo_final, X_train_feat)
    erro_treino = calcular_erro_classificacao(y_train.astype(np.int64), y_pred_train.astype(np.int64))
    print(f"\n[Resultado] Erro no conjunto de TREINO (melhor modelo): {erro_treino:.4f} | Acurácia: {1.0-erro_treino:.4f}")

    # Diagnóstico pós-treino
    diagnostico_modelo(
        modelo=modelo_final,
        y_train=y_train.astype(np.int64),
        y_pred_train=y_pred_train.astype(np.int64),
        logits_train=logits_train,
        titulo="Diagnóstico PÓS-TREINO (modelo final)",
        max_amostras_prob=CONFIG["diag_max_amostras_prob"],
        seed=CONFIG["diag_seed"]
    )

    # Exemplos de previsão
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

if __name__ == "__main__":
    main()
