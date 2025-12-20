import joblib
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import StratifiedKFold

# (Opcional) suporte a sparse sem quebrar se scipy não estiver instalado
try:
    from scipy import sparse  # type: ignore
except Exception:
    sparse = None

# ============================================================
# CONFIG (ajuste aqui para prototipagem rápida)
# ============================================================

CONFIG = {
    # Dataset
    "dataset_filename": "celeba_hog_128x128_o9.joblib",

    # Seleção de classes ANTES do split (prototipagem)
    "frac_classes": 0.05,          # ex: 0.05 = usa 5% das classes aleatórias
    "seed_classes": 42,

    # Split treino/teste (sem validação)
    "test_frac": 0.09,
    "seed_split": 42,
    "n_min_treino_por_classe": 1,  # garante pelo menos 1 amostra por classe no treino

    # Random Projection (sobre o HOG)
    # Dica: comece com 256 ou 512 pra acelerar MUITO.
    "rp_n_components": 512,
    "rp_seed": 42,

    # CV
    "k_folds": 3,
    "seed_cv": 42,

    # Grid de lambdas (Elastic Net) -> ajuste livremente
    # Dica: grid pequeno para testar; depois amplia.
    "lambda_l1_grid": [0.0, 0.05, 0.1, 0.5],
    "lambda_l2_grid": [0.0, 0.05, 0.1, 0.5],

    # Treino por GD (use menos épocas no CV; mais no final)
    "epocas_cv": 15,
    "epocas_final": 30,
    "taxa_aprendizado": 0.05,
    "tamanho_lote": 256,

    # Progresso do GD (evita print a cada lote, que deixa lento)
    "imprimir_a_cada_n_lotes": 25,

    # Output
    "n_exemplos_previsao": 10,
    "exemplos_de": "teste",  # "teste" ou "treino"
}

# ============================================================
# Utilitários
# ============================================================

def l2_normalize_rows(X, eps: float = 1e-12):
    """
    Normaliza cada amostra (linha) para norma L2=1.
    Funciona com np.ndarray e (se scipy estiver disponível) com matrizes esparsas.
    """
    if (sparse is not None) and sparse.issparse(X):
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        norms = np.maximum(norms, eps)
        inv = 1.0 / norms
        return X.multiply(inv[:, None])
    else:
        X = np.asarray(X)
        # evita divisão por zero
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return X / norms

def codificar_rotulos_com_classes(y: np.ndarray, classes: np.ndarray):
    """
    Mapeia rótulos -> índices 0..K-1 usando um vetor 'classes' fixo.
    """
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

# ============================================================
# Seleção de classes ANTES do split
# ============================================================

def selecionar_classes_aleatorias(X, y, frac_classes: float, seed: int, min_amostras_por_classe: int):
    """
    1) escolhe aleatoriamente uma % das classes
    2) filtra dataset para só essas classes
    3) remove classes que ficaram com poucas amostras (min_amostras_por_classe)
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_classes_total = len(classes)

    n_escolher = max(1, int(np.round(frac_classes * n_classes_total)))
    escolhidas = rng.choice(classes, size=n_escolher, replace=False)

    mask = np.isin(y, escolhidas)
    X2, y2 = X[mask], y[mask]

    # remove classes com poucas amostras
    c2, counts = np.unique(y2, return_counts=True)
    classes_ok = c2[counts >= min_amostras_por_classe]
    mask2 = np.isin(y2, classes_ok)

    X3, y3 = X2[mask2], y2[mask2]

    return X3, y3, classes_ok

# ============================================================
# Split garantindo que TODAS as classes tenham amostra no treino
# ============================================================

def split_garantindo_treino_por_classe(X, y, test_frac: float, seed: int, n_min_treino_por_classe: int = 1):
    """
    Faz split treino/teste garantindo pelo menos n_min_treino_por_classe por classe no treino.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    test_n = int(np.round(test_frac * n))

    # agrupa índices por classe via sorting (mais eficiente que vários np.where)
    order = np.argsort(y)
    y_sorted = y[order]
    classes, start_idx = np.unique(y_sorted, return_index=True)
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
    # o resto vai para treino
    test_set = set(test_idx.tolist())
    resto_para_treino = np.array([i for i in resto_idx if i not in test_set], dtype=np.int64)

    train_idx = np.concatenate([train_idx, resto_para_treino])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ============================================================
# Gradiente descendente com progresso (throttle)
# ============================================================

def gradiente_descendente(
    params_iniciais: dict,
    funcao_perda_grad,
    X: np.ndarray,
    Y: np.ndarray,
    taxa_aprendizado: float,
    epocas: int,
    tamanho_lote: int | None,
    seed: int,
    verbose: bool,
    imprimir_a_cada_n_lotes: int = 25
):
    """
    GD com passo escolhido por line search (backtracking) usando condição de Armijo.
    Mantém a mesma API do seu script: taxa_aprendizado é o alpha inicial.
    """
    rng = np.random.default_rng(seed)

    # ---- Parâmetros do Armijo (fixos aqui para não alterar CONFIG / API) ----
    c = 1e-4          # parâmetro Armijo (suficiente diminuição)
    rho = 0.5         # fator de redução do passo (backtracking)
    max_backtracks = 12
    alpha_min = 1e-10
    # ------------------------------------------------------------------------

    params = {}
    for k, v in params_iniciais.items():
        params[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v

    n = X.shape[0]
    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n:
        tamanho_lote = n

    n_lotes = int(np.ceil(n / tamanho_lote))
    historico = []

    for epoca in range(epocas):
        idx = rng.permutation(n)

        for num_lote, ini in enumerate(range(0, n, tamanho_lote), start=1):
            lote = idx[ini:ini + tamanho_lote]
            Xb = X[lote]
            Yb = Y[lote]

            # perda e gradiente no ponto atual
            perda0, grads = funcao_perda_grad(params, Xb, Yb)

            # norma^2 do gradiente (somando todos os parâmetros)
            grad_norm_sq = 0.0
            for nome_param, g in grads.items():
                grad_norm_sq += float(np.sum(g * g))

            # se gradiente ~0, não atualiza
            if grad_norm_sq <= 0.0:
                continue

            # direção de descida p = -grad
            alpha = float(taxa_aprendizado)  # alpha inicial (o seu antigo LR fixo)

            # backtracking Armijo
            aceitou = False
            while True:
                # params_candidate = params - alpha * grad
                params_cand = {}
                for nome_param in params:
                    if nome_param in grads:
                        params_cand[nome_param] = params[nome_param] - alpha * grads[nome_param]
                    else:
                        params_cand[nome_param] = params[nome_param]

                perda_cand, _ = funcao_perda_grad(params_cand, Xb, Yb)

                # Condição de Armijo:
                # f(x + alpha*p) <= f(x) + c*alpha*<grad, p>
                # com p = -grad => <grad,p> = -||grad||^2
                if perda_cand <= perda0 - c * alpha * grad_norm_sq:
                    aceitou = True
                    params = params_cand
                    break

                alpha *= rho
                if alpha < alpha_min:
                    break
                if max_backtracks <= 0:
                    break
                max_backtracks -= 1

            # (opcional) se não aceitou, ainda assim dá um passo minúsculo (evita travar total)
            if not aceitou:
                alpha = max(alpha, alpha_min)
                for nome_param in grads:
                    params[nome_param] = params[nome_param] - alpha * grads[nome_param]

            # progresso
            if (num_lote % imprimir_a_cada_n_lotes == 0) or (num_lote == n_lotes):
                print(f"[Treino] Época {epoca+1}/{epocas} | Lote {num_lote}/{n_lotes}", end="\r")

        print(" " * 90, end="\r")
        if verbose:
            perda_full, _ = funcao_perda_grad(params, X, Y)
            historico.append(float(perda_full))
            print(f"[GD+Armijo] Época {epoca+1}/{epocas} | perda={perda_full:.6f}")

    return params, historico


# ============================================================
# Regressão Linear Multiclasse (MSE vs one-hot) + Elastic Net
# ============================================================

def perda_e_gradiente_regressao_linear_elasticnet(params, X, Y, lambda_l1, lambda_l2):
    W = params["W"]  # (K, d)
    b = params["b"]  # (K,)

    m, d = X.shape
    K = Y.shape[1]

    S = X @ W.T + b        # (m, K)
    E = S - Y              # (m, K)
    mse = np.mean(E * E)

    reg = lambda_l1 * np.sum(np.abs(W)) + 0.5 * lambda_l2 * np.sum(W * W)
    perda_total = mse + reg

    fator = 2.0 / (m * K)
    dS = fator * E

    grad_W = dS.T @ X
    grad_b = np.sum(dS, axis=0)

    grad_W += lambda_l2 * W
    grad_W += lambda_l1 * np.sign(W)

    return float(perda_total), {"W": grad_W, "b": grad_b}

def treinar_regressao_linear_multiclasse_elasticnet(
    X_treino, y_treino, classes_fixas,
    lambda_l1, lambda_l2,
    taxa_aprendizado, epocas, tamanho_lote,
    seed, verbose, imprimir_a_cada_n_lotes
):
    y_idx, classes, mapa = codificar_rotulos_com_classes(y_treino, classes_fixas)
    K = len(classes)
    d = X_treino.shape[1]
    Y = one_hot(y_idx, K)

    rng = np.random.default_rng(seed)
    W0 = rng.normal(0.0, 0.01, size=(K, d)).astype(np.float64)
    b0 = np.zeros(K, dtype=np.float64)
    params0 = {"W": W0, "b": b0}

    def f(params, Xb, Yb):
        return perda_e_gradiente_regressao_linear_elasticnet(
            params,
            Xb.astype(np.float64, copy=False),
            Yb,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2
        )

    params_finais, historico = gradiente_descendente(
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
        "mapa": mapa,
        "historico_perda": historico,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2
    }

def prever_regressao_linear_multiclasse(modelo, X):
    W, b, classes = modelo["W"], modelo["b"], modelo["classes"]
    scores = X.astype(np.float64, copy=False) @ W.T + b
    idx_pred = np.argmax(scores, axis=1)
    return classes[idx_pred], scores

# ============================================================
# K-Fold CV para escolher lambdas
# ============================================================

def escolher_melhores_lambdas_por_cv(
    X_train, y_train, classes_fixas,
    lambda_l1_grid, lambda_l2_grid,
    k_folds, seed_cv,
    treino_cfg
):
    # ✅ Stratified K-Fold (você já estava usando; mantido)
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
                imprimir_a_cada_n_lotes=treino_cfg["imprimir_a_cada_n_lotes"]
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

    return melhor  # (mean_err, l1, l2)

# ============================================================
# PIPELINE PRINCIPAL
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

    # ✅ (1) Normalização L2 pós-HOG (por amostra)
    print("\n[Etapa 0/5] Normalizando HOG (L2 por amostra)...")
    X = l2_normalize_rows(X)

    # 1) Seleção de classes ANTES do split
    k = CONFIG["k_folds"]
    min_amostras_por_classe = max(k, 2)  # garante CV viável

    print(f"\n[Etapa 1/5] Selecionando {CONFIG['frac_classes']*100:.1f}% das classes (seed={CONFIG['seed_classes']})...")
    X, y, classes_ok = selecionar_classes_aleatorias(
        X, y,
        frac_classes=CONFIG["frac_classes"],
        seed=CONFIG["seed_classes"],
        min_amostras_por_classe=min_amostras_por_classe
    )
    print("[Etapa 1/5] Após filtro:")
    print("  X:", X.shape, " | n classes:", len(np.unique(y)))

    # 2) Split treino/teste sem validação
    print(f"\n[Etapa 2/5] Split treino/teste (test_frac={CONFIG['test_frac']:.2f}) garantindo 1+ amostra/classe no treino...")
    X_train, X_test, y_train, y_test = split_garantindo_treino_por_classe(
        X, y,
        test_frac=CONFIG["test_frac"],
        seed=CONFIG["seed_split"],
        n_min_treino_por_classe=CONFIG["n_min_treino_por_classe"]
    )
    print("[Etapa 2/5] Shapes:")
    print("  Train:", X_train.shape, y_train.shape)
    print("  Test :", X_test.shape,  y_test.shape)

    # 3) Random Projection (fit no treino; transforma treino e teste)
    print(f"\n[Etapa 3/5] Random Projection: n_components={CONFIG['rp_n_components']} (seed={CONFIG['rp_seed']})")
    rp = SparseRandomProjection(n_components=CONFIG["rp_n_components"], random_state=CONFIG["rp_seed"])
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp  = rp.transform(X_test)
    print("[Etapa 3/5] Shapes após RP:")
    print("  Train RP:", X_train_rp.shape)
    print("  Test  RP:", X_test_rp.shape)

    # ✅ (1) Normalização L2 pós-RP (por amostra)
    print("[Etapa 3/5] Normalizando pós-RP (L2 por amostra)...")
    X_train_rp = l2_normalize_rows(X_train_rp)
    X_test_rp  = l2_normalize_rows(X_test_rp)

    # classes fixas para manter o mesmo espaço de saída (K) em todos os folds
    classes_fixas = np.unique(y_train)

    # 4) K-Fold CV para escolher lambdas
    print(f"\n[Etapa 4/5] K-Fold CV (k={CONFIG['k_folds']}) para escolher lambdas...")
    treino_cfg = {
        "taxa_aprendizado": CONFIG["taxa_aprendizado"],
        "epocas_cv": CONFIG["epocas_cv"],
        "tamanho_lote": CONFIG["tamanho_lote"],
        "seed_treino": 123,
        "imprimir_a_cada_n_lotes": CONFIG["imprimir_a_cada_n_lotes"]
    }

    best = escolher_melhores_lambdas_por_cv(
        X_train_rp, y_train, classes_fixas,
        CONFIG["lambda_l1_grid"], CONFIG["lambda_l2_grid"],
        k_folds=CONFIG["k_folds"],
        seed_cv=CONFIG["seed_cv"],
        treino_cfg=treino_cfg
    )
    best_mean_err, best_l1, best_l2 = best
    print(f"\n[CV] Melhor combinação final: mean_err={best_mean_err:.4f} | lambda_l1={best_l1} | lambda_l2={best_l2}")

    # 5) Treina modelo final com os melhores lambdas (no treino inteiro)
    print(f"\n[Etapa 5/5] Treinando modelo final no treino inteiro (epocas={CONFIG['epocas_final']})...")
    modelo_final = treinar_regressao_linear_multiclasse_elasticnet(
        X_train_rp, y_train, classes_fixas,
        lambda_l1=best_l1,
        lambda_l2=best_l2,
        taxa_aprendizado=CONFIG["taxa_aprendizado"],
        epocas=CONFIG["epocas_final"],
        tamanho_lote=CONFIG["tamanho_lote"],
        seed=999,
        verbose=True,
        imprimir_a_cada_n_lotes=CONFIG["imprimir_a_cada_n_lotes"]
    )

    # Output pedido: erro no TREINO do melhor modelo
    y_pred_train, _ = prever_regressao_linear_multiclasse(modelo_final, X_train_rp)
    erro_treino = calcular_erro_classificacao(y_train.astype(np.int64), y_pred_train.astype(np.int64))
    print(f"\n[Resultado] Erro no conjunto de TREINO (melhor modelo): {erro_treino:.4f} | Acurácia: {1.0-erro_treino:.4f}")

    # 10 exemplos de previsão (por padrão: do teste)
    if CONFIG["exemplos_de"].lower() == "treino":
        y_true_ex, y_pred_ex = y_train, y_pred_train
    else:
        y_pred_test, _ = prever_regressao_linear_multiclasse(modelo_final, X_test_rp)
        y_true_ex, y_pred_ex = y_test, y_pred_test

    mostrar_previsoes_amostrais(
        y_true_ex.astype(np.int64),
        y_pred_ex.astype(np.int64),
        n_amostras=CONFIG["n_exemplos_previsao"],
        seed=CONFIG["seed_split"]
    )

if __name__ == "__main__":
    main()
