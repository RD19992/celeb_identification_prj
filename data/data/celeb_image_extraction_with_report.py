# -*- coding: utf-8 -*-
"""
CELEBA HOG -> Softmax Regression (do zero, GD + Armijo) com CV estratificado.

CORREÇÕES IMPORTANTES NESTA VERSÃO:
1) [FIX CV 4 MEMBROS] A amostragem para CV agora GARANTE >= k amostras por classe,
   evitando o warning do StratifiedKFold (classe com 4 membros quando k=5).
   - Antes: amostrava classes até passar do alvo e depois "cortava" (truncate) globalmente,
     podendo remover 1 amostra de uma classe e deixá-la com k-1.
   - Agora: escolhe N_classes = floor(alvo / k) e pega exatamente k por classe (sem truncate global).
2) [FIX MÉTRICAS/EXEMPLOS] O erro/acurácia reportados para TREINO e TESTE são calculados
   usando o MESMO y_pred que vai para os exemplos. Também há um "sanity check" que imprime
   quantos erros existem e mostra alguns erros reais se houver inconsistência.

Observação: mantive o "truque padrão" (sem one-hot global). O código trabalha com:
- y_idx (0..K-1) internamente para gradiente/loss
- y_labels (rótulos originais) para métricas/report.

Requisitos:
- joblib (para carregar X,y do dataset)
- numpy
- scikit-learn (train_test_split, StratifiedKFold, confusion_matrix)
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
    "frac_classes": 0.005,           # fração das CLASSES ELEGÍVEIS (ex: 0.005 = 0.5%)
    "seed_classes": 42,
    "min_amostras_por_classe": 25,   # mínimo no dataset inteiro para classe ser elegível

    # split treino/teste
    "test_frac": 0.09,
    "seed_split": 42,
    "min_train_por_classe": 5,       # mínimo no TREINO (após split) p/ manter classe

    # CV
    "k_folds": 5,                    # ajuste aqui (ex: 3, 5)
    "cv_frac": 0.05,                 # [ALTERAÇÃO pedida anteriormente] 5% do treino p/ CV
    "final_frac": 0.20,              # [ALTERAÇÃO pedida anteriormente] 20% do treino p/ treino final

    # treinamento
    "epochs": 80,
    "alpha_init": 2.0,               # [ALTERAÇÃO] alpha inicial maior (Armijo vai reduzir se precisar)
    "armijo_beta": 0.5,              # fator de backtracking
    "armijo_sigma": 1e-4,            # parâmetro de suficiência (Armijo)
    "eps_std": 1e-6,

    # grid de regularização (Elastic Net no W, sem regularizar bias)
    # loss = CE + (l2/(2m))*||W||^2 + (l1/m)*||W||_1
    "grid_l1": [0.0, 1e-4, 3e-4, 1e-3],
    "grid_l2": [0.0, 1e-4, 3e-4, 1e-3],

    # outputs
    "n_exemplos_previsao": 10,
    "top_k_confusao": 10,
}


# ============================================================
# Utils: carregamento / checks
# ============================================================

def carregar_dataset_joblib(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "X" in obj and "y" in obj:
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
# Seleção de classes / filtros
# ============================================================

def filtrar_classes_min_amostras(X, y, min_amostras: int):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    elegiveis = classes[counts >= min_amostras]
    mask = np.isin(y, elegiveis)
    return X[mask], y[mask], elegiveis


def selecionar_frac_classes(classes_elegiveis: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(classes_elegiveis)
    k = max(1, int(np.round(frac * n)))
    idx = np.arange(n)
    rng.shuffle(idx)
    escolhidas = classes_elegiveis[idx[:k]]
    return np.array(escolhidas, dtype=np.int64)


def filtrar_por_classes(X, y, classes):
    y = np.asarray(y)
    mask = np.isin(y, classes)
    return X[mask], y[mask]


def garantir_min_treino_por_classe(X_train, y_train, X_test, y_test, min_train_por_classe: int):
    # remove classes que no treino ficaram com < min_train_por_classe
    classes, counts = np.unique(y_train, return_counts=True)
    ok = classes[counts >= min_train_por_classe]
    X_train2, y_train2 = filtrar_por_classes(X_train, y_train, ok)
    X_test2, y_test2 = filtrar_por_classes(X_test, y_test, ok)
    return X_train2, y_train2, X_test2, y_test2, ok


# ============================================================
# Padronização (somente mean/std do TREINO) - float32
# ============================================================

def fit_standardizer(X_train: np.ndarray, eps: float):
    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    return mean, std, float(eps)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float):
    return (X - mean) / (std + eps)


# ============================================================
# Amostragem estratificada
# ============================================================

def amostrar_estratificado(y: np.ndarray, frac: float, seed: int):
    """Retorna índices de uma amostra estratificada (proporcional) com tamanho ~ frac*N."""
    y = np.asarray(y)
    N = len(y)
    n = max(1, int(np.round(frac * N)))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(splitter.split(np.zeros(N), y))
    return idx.astype(np.int64, copy=False)


def amostrar_para_cv_por_classes(y_train: np.ndarray, frac: float, seed: int, min_por_classe: int):
    """
    [FIX CV 4 MEMBROS] Amostra para CV garantindo no mínimo min_por_classe por classe.

    - alvo = round(frac*N)
    - n_classes_escolhidas = min( classes_ok, floor(alvo / min_por_classe) ), pelo menos 1
    - retorna exatamente min_por_classe amostras por classe escolhida
    """
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)

    N = int(len(y_train))
    alvo = int(np.round(frac * N))
    alvo = max(min_por_classe, alvo)
    if alvo <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    classes, counts = np.unique(y_train, return_counts=True)
    classes_ok = classes[counts >= min_por_classe]
    if len(classes_ok) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # quantas classes dá para garantir min_por_classe sem "truncate global"
    max_classes = max(1, alvo // min_por_classe)
    n_classes = min(len(classes_ok), max_classes)

    # se n_classes * min_por_classe == 0, aborta
    if n_classes <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rng.shuffle(classes_ok)
    classes_escolhidas = classes_ok[:n_classes].astype(np.int64, copy=False)

    amostra_idx = []
    for c in classes_escolhidas:
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)
        amostra_idx.append(idx_c[:min_por_classe])

    amostra_idx = np.concatenate(amostra_idx).astype(np.int64, copy=False)
    # não fazemos truncate aqui, por desenho.
    return amostra_idx, classes_escolhidas


# ============================================================
# "Truque padrão": codificar rótulos p/ índices 0..K-1
# ============================================================

def codificar_rotulos_com_classes(y_labels: np.ndarray, classes: np.ndarray):
    classes = np.asarray(classes, dtype=np.int64)
    mapa = {int(c): i for i, c in enumerate(classes)}
    y_idx = np.array([mapa[int(v)] for v in y_labels], dtype=np.int64)
    return y_idx, classes, mapa


# ============================================================
# Softmax regression: loss/grad/predict - float32
# ============================================================

def softmax_stable(Z: np.ndarray):
    # Z: (N,K)
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z, dtype=np.float64)  # estabilidade
    P = expZ / expZ.sum(axis=1, keepdims=True)
    return P.astype(np.float64, copy=False)


def loss_ce_elasticnet(X: np.ndarray, y_idx: np.ndarray, W: np.ndarray, b: np.ndarray, l1: float, l2: float):
    """
    loss = CE + (l2/(2m))*||W||^2 + (l1/m)*||W||_1
    """
    m = X.shape[0]
    logits = (X @ W + b).astype(np.float64, copy=False)
    P = softmax_stable(logits)  # float64
    # CE
    logp = -np.log(P[np.arange(m), y_idx] + 1e-12)
    ce = float(np.mean(logp))
    # reg (escala /m)
    reg_l2 = (l2 / (2.0 * m)) * float(np.sum((W.astype(np.float64)) ** 2))
    reg_l1 = (l1 / m) * float(np.sum(np.abs(W.astype(np.float64))))
    return ce + reg_l2 + reg_l1


def grad_ce_elasticnet(X: np.ndarray, y_idx: np.ndarray, W: np.ndarray, b: np.ndarray, l1: float, l2: float):
    """
    Gradiente do loss acima. Sem regularizar b.
    """
    m, d = X.shape
    K = W.shape[1]

    logits = (X @ W + b).astype(np.float64, copy=False)  # (m,K)
    P = softmax_stable(logits)  # float64
    P[np.arange(m), y_idx] -= 1.0  # P - Y_onehot

    # grad
    dW = (X.T.astype(np.float64) @ P) / m
    db = P.mean(axis=0)

    # reg (escala /m)
    if l2 != 0.0:
        dW += (l2 / m) * W.astype(np.float64)
    if l1 != 0.0:
        dW += (l1 / m) * np.sign(W.astype(np.float64))

    return dW.astype(np.float32), db.astype(np.float32)


def predict_logits(X: np.ndarray, W: np.ndarray, b: np.ndarray):
    return (X @ W + b).astype(np.float32, copy=False)


def predict_labels(X: np.ndarray, W: np.ndarray, b: np.ndarray, classes: np.ndarray):
    logits = predict_logits(X, W, b)  # float32
    pred_idx = np.argmax(logits, axis=1).astype(np.int64)
    return classes[pred_idx], logits


# ============================================================
# Treino por GD + Armijo
# ============================================================

def treinar_softmax_armijo(
    X_train: np.ndarray,
    y_train_labels: np.ndarray,
    classes_fixas: np.ndarray,
    l1: float,
    l2: float,
    epochs: int,
    alpha_init: float,
    beta: float,
    sigma: float,
    seed: int = 0,
):
    """
    Treina modelo (W,b) com Armijo. Retorna dict com classes, W, b e histórico de alphas aceitos.
    """
    rng = np.random.default_rng(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train_labels = np.asarray(y_train_labels, dtype=np.int64)
    classes_fixas = np.asarray(classes_fixas, dtype=np.int64)

    y_idx, classes_fixas, _ = codificar_rotulos_com_classes(y_train_labels, classes_fixas)

    m, d = X_train.shape
    K = len(classes_fixas)

    W = (0.01 * rng.standard_normal((d, K))).astype(np.float32)
    b = np.zeros((K,), dtype=np.float32)

    alphas_aceitos = []

    # embaralhamento por época
    for ep in range(epochs):
        perm = rng.permutation(m)
        Xb = X_train[perm]
        yb = y_idx[perm]

        # grad em batch único (pode ser estendido para mini-batch)
        dW, db = grad_ce_elasticnet(Xb, yb, W, b, l1=l1, l2=l2)

        # Armijo line search
        alpha = float(alpha_init)
        loss0 = loss_ce_elasticnet(Xb, yb, W, b, l1=l1, l2=l2)
        grad_norm2 = float(np.sum(dW.astype(np.float64) ** 2) + np.sum(db.astype(np.float64) ** 2))

        # backtracking
        while True:
            W_new = (W - alpha * dW).astype(np.float32)
            b_new = (b - alpha * db).astype(np.float32)
            loss_new = loss_ce_elasticnet(Xb, yb, W_new, b_new, l1=l1, l2=l2)

            if loss_new <= loss0 - sigma * alpha * grad_norm2:
                W, b = W_new, b_new
                alphas_aceitos.append(alpha)
                break

            alpha *= beta
            if alpha < 1e-10:
                # passo ficou minúsculo: aceita mesmo assim para não travar
                W, b = W_new, b_new
                alphas_aceitos.append(alpha)
                break

        if (ep + 1) % 10 == 0 or ep == 0:
            # log leve
            y_pred, _ = predict_labels(X_train, W, b, classes_fixas)
            acc = float(np.mean(y_pred == y_train_labels))
            print(f"    [Treino ep {ep+1:03d}/{epochs}] loss~{loss0:.4f} | alpha={alpha:.3e} | acc={acc:.4f}")

    return {"classes": classes_fixas, "W": W, "b": b, "alphas": np.array(alphas_aceitos, dtype=np.float64)}


# ============================================================
# Avaliação / diagnóstico
# ============================================================

def report_metrics_and_examples(nome: str, y_true: np.ndarray, y_pred: np.ndarray, n_exemplos: int, seed: int):
    """
    [FIX MÉTRICAS/EXEMPLOS] Calcula erro/acurácia e imprime exemplos a partir do MESMO y_pred.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) != len(y_pred):
        print(f"\n[{nome}] ERRO: tamanhos diferentes y_true={len(y_true)} vs y_pred={len(y_pred)}")
        return

    err = float(np.mean(y_true != y_pred))
    acc = 1.0 - err
    n_err = int(np.sum(y_true != y_pred))

    print(f"\n[{nome}] Erro={err:.4f} | Acurácia={acc:.4f} | N={len(y_true)} | n_err={n_err}")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    idxs = rng.choice(n, size=min(n_exemplos, n), replace=False)

    print(f"[{nome}] Exemplos (verdadeiro -> predito)  (✓=certo, ✗=erro)")
    for j, i in enumerate(idxs, 1):
        ok = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"  #{j:02d} [{ok}] {int(y_true[i])} -> {int(y_pred[i])}")

    # sanity: se o usuário achar inconsistente, imprimimos alguns erros reais
    if n_err > 0:
        idx_err = np.where(y_true != y_pred)[0]
        k = min(5, len(idx_err))
        print(f"[{nome}] Primeiros {k} erros reais (index: true -> pred):")
        for i in idx_err[:k]:
            print(f"   - {int(i)}: {int(y_true[i])} -> {int(y_pred[i])}")


def avaliar_teste_top10(y_test: np.ndarray, y_pred_test: np.ndarray, top_k: int = 10):
    y_test = np.asarray(y_test, dtype=np.int64)
    y_pred_test = np.asarray(y_pred_test, dtype=np.int64)

    classes, counts = np.unique(y_test, return_counts=True)
    ordem = np.argsort(-counts)
    top = classes[ordem[:top_k]]

    mask = np.isin(y_test, top)
    y_t = y_test[mask]
    y_p = y_pred_test[mask]

    labels = top  # ordem fixa (top mais comuns)
    cm = confusion_matrix(y_t, y_p, labels=labels)

    print(f"\n[Matriz de Confusão] Top-{top_k} classes mais comuns no TESTE")
    print("Labels (classe):", labels.tolist())
    print(cm)


# ============================================================
# CV: escolher lambdas (grid) via StratifiedKFold
# ============================================================

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
):
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    best = None
    best_score = -1.0
    best_alpha_med = None

    for l1 in grid_l1:
        for l2 in grid_l2:
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
                    epochs=epochs, alpha_init=alpha_init, beta=beta, sigma=sigma,
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
    path = CONFIG["dataset_path"]
    print(f"Dataset path: {path}")
    print(f"Exists? {Path(path).exists()}")

    X, y = carregar_dataset_joblib(path)

    # [ALTERAÇÃO] usar sempre float32
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print("\n[Info] Dataset original")
    print(f"X: {X.shape} {X.dtype}")
    print(f"y: {y.shape} {y.dtype}")
    print(f"n classes: {len(np.unique(y))}")

    diagnostico_hog_dim(X.shape[1])

    # 1) filtra classes elegíveis
    print(f"\n[Etapa 1/7] Filtrando classes com >= {CONFIG['min_amostras_por_classe']} amostras...")
    Xf, yf, classes_elegiveis = filtrar_classes_min_amostras(X, y, CONFIG["min_amostras_por_classe"])
    print(f"[Etapa 1/7] Após filtro: X={Xf.shape} | n classes elegíveis={len(classes_elegiveis)}")

    # 2) escolhe fração das classes
    print(f"\n[Etapa 2/7] Selecionando {100*CONFIG['frac_classes']:.3f}% das classes elegíveis...")
    classes_sel = selecionar_frac_classes(classes_elegiveis, CONFIG["frac_classes"], CONFIG["seed_classes"])
    Xs, ys = filtrar_por_classes(Xf, yf, classes_sel)
    print(f"[Etapa 2/7] Após seleção: X={Xs.shape} | n classes={len(np.unique(ys))}")

    if Xs.shape[0] == 0:
        raise RuntimeError("Seleção de classes resultou em dataset vazio. Ajuste frac_classes/min_amostras.")

    # 3) split treino/teste
    print(f"\n[Etapa 3/7] Split treino/teste (test_frac={CONFIG['test_frac']})...")
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=CONFIG["test_frac"], random_state=CONFIG["seed_split"], stratify=ys
    )
    print(f"[Etapa 3/7] Train: {X_train.shape} | Test: {X_test.shape}")

    # 4) garante mínimo no treino para manter classes
    min_train = max(CONFIG["min_train_por_classe"], CONFIG["k_folds"])
    print(f"\n[Etapa 4/7] Garantindo mínimo {min_train} amostras/classe no TREINO (p/ CV estratificado)...")
    X_train, y_train, X_test, y_test, classes_ok = garantir_min_treino_por_classe(
        X_train, y_train, X_test, y_test, min_train_por_classe=min_train
    )
    print(f"[Etapa 4/7] Após filtro: Train={X_train.shape} (classes={len(np.unique(y_train))}) | Test={X_test.shape}")

    if X_train.shape[0] == 0 or len(np.unique(y_train)) < 2:
        raise RuntimeError("Treino ficou vazio ou com <2 classes após filtros. Ajuste parâmetros.")

    # 5) padroniza usando somente treino
    print("\n[Etapa 5/7] Padronizando: (X - mean_train) / (std_train + eps) ...")
    mean, std, eps = fit_standardizer(X_train, CONFIG["eps_std"])
    X_train_feat = apply_standardizer(X_train, mean, std, eps).astype(np.float32)
    X_test_feat = apply_standardizer(X_test, mean, std, eps).astype(np.float32)
    print(f"[Etapa 5/7] OK. mean/std dtype: {mean.dtype}/{std.dtype}")

    # fixa classes (ordem) do modelo: usa classes presentes no treino
    classes_fixas = np.unique(y_train).astype(np.int64)

    # 6) amostra para CV garantindo >= k por classe
    print(f"\n[Etapa 6/7] Amostrando para CV (frac={CONFIG['cv_frac']}, k={CONFIG['k_folds']}) ...")
    idx_cv, classes_cv = amostrar_para_cv_por_classes(
        y_train, frac=CONFIG["cv_frac"], seed=CONFIG["seed_split"], min_por_classe=CONFIG["k_folds"]
    )
    if idx_cv.size == 0:
        raise RuntimeError("Amostra para CV ficou vazia. Ajuste cv_frac/min_amostras/k_folds.")
    X_cv = X_train_feat[idx_cv]
    y_cv = y_train[idx_cv]
    print(f"[Etapa 6/7] CV sample: X={X_cv.shape} | classes={len(np.unique(y_cv))}")
    # sanity: deve garantir >=k
    _, cts = np.unique(y_cv, return_counts=True)
    print(f"[Etapa 6/7] min count por classe no CV sample = {int(cts.min())} (deve ser >= {CONFIG['k_folds']})")

    # 6.5) escolher lambdas por CV
    print("\n[Etapa 6.5/7] Rodando grid-search por CV...")
    (best_l1, best_l2), best_score, best_alpha_med = escolher_melhores_lambdas_por_cv(
        X_train_feat=X_cv,
        y_train_labels=y_cv,
        classes_fixas=np.unique(y_cv).astype(np.int64),  # classes presentes no CV sample
        k_folds=CONFIG["k_folds"],
        grid_l1=CONFIG["grid_l1"],
        grid_l2=CONFIG["grid_l2"],
        epochs=CONFIG["epochs"],
        alpha_init=CONFIG["alpha_init"],
        beta=CONFIG["armijo_beta"],
        sigma=CONFIG["armijo_sigma"],
        seed=CONFIG["seed_split"],
    )
    print(f"\n[CV] Melhor: l1={best_l1} | l2={best_l2} | mean_acc={best_score:.4f} | alpha_mediana~{best_alpha_med:.3e}")

    # 7) treino final em amostra maior (final_frac) e avaliação no teste
    print(f"\n[Etapa 7/7] Treino final em {100*CONFIG['final_frac']:.1f}% do treino...")
    idx_final = amostrar_estratificado(y_train, frac=CONFIG["final_frac"], seed=CONFIG["seed_split"])
    X_final = X_train_feat[idx_final]
    y_final = y_train[idx_final]
    classes_final = np.unique(y_final).astype(np.int64)

    print(f"[Etapa 7/7] Final sample: X={X_final.shape} | classes={len(classes_final)}")

    modelo_final = treinar_softmax_armijo(
        X_final, y_final, classes_fixas=classes_final,
        l1=float(best_l1), l2=float(best_l2),
        epochs=CONFIG["epochs"],
        alpha_init=float(best_alpha_med) if best_alpha_med is not None else float(CONFIG["alpha_init"]),
        beta=CONFIG["armijo_beta"], sigma=CONFIG["armijo_sigma"],
        seed=CONFIG["seed_split"],
    )

    y_pred_train, _ = predict_labels(X_final, modelo_final["W"], modelo_final["b"], modelo_final["classes"])
    y_pred_test, _ = predict_labels(X_test_feat, modelo_final["W"], modelo_final["b"], modelo_final["classes"])

    # [FIX] métricas + exemplos consistentes
    report_metrics_and_examples("TREINO (final sample)", y_final, y_pred_train, CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])
    report_metrics_and_examples("TESTE", y_test, y_pred_test, CONFIG["n_exemplos_previsao"], seed=CONFIG["seed_split"])

    # matriz confusão top-k
    avaliar_teste_top10(y_test=y_test, y_pred_test=y_pred_test, top_k=CONFIG["top_k_confusao"])

# ============================================================
# Relatórios rápidos de distribuição de classes (top/bottom)
# Cole este bloco antes do `if __name__ == "__main__":`
# ============================================================

def _print_top_bottom_from_counts(classes_sorted, counts_sorted, top_n=10, bottom_n=10, titulo=""):
    total = int(np.sum(counts_sorted))
    n_classes = int(len(classes_sorted))

    if titulo:
        print(f"\n[{titulo}]")

    print(f"  Total de instâncias no subset: {total}")
    print(f"  Total de classes no subset:    {n_classes}")

    if n_classes == 0:
        print("  (subset vazio)")
        return

    # Top-N
    k_top = min(top_n, n_classes)
    print(f"\n  Top-{k_top} classes (mais frequentes):")
    for i in range(k_top):
        c = int(classes_sorted[i])
        cnt = int(counts_sorted[i])
        pct = 100.0 * cnt / max(1, total)
        print(f"    #{i+1:02d}  classe={c}  n={cnt}  ({pct:.2f}%)")

    # Bottom-N
    k_bot = min(bottom_n, n_classes)
    print(f"\n  Bottom-{k_bot} classes (menos frequentes):")
    start = n_classes - k_bot
    for j in range(k_bot):
        i = start + j
        c = int(classes_sorted[i])
        cnt = int(counts_sorted[i])
        pct = 100.0 * cnt / max(1, total)
        print(f"    #{j+1:02d}  classe={c}  n={cnt}  ({pct:.2f}%)")


def relatorio_top_bottom_classes(y_subset, top_n=10, bottom_n=10, titulo=""):
    """
    Dado um y_subset (ex.: y_final), imprime quantas instâncias existem
    nas top-10 e bottom-10 classes desse subset.
    """
    y_subset = np.asarray(y_subset)
    classes, counts = np.unique(y_subset, return_counts=True)
    ordem = np.argsort(-counts)  # desc
    classes_sorted = classes[ordem]
    counts_sorted = counts[ordem]
    _print_top_bottom_from_counts(classes_sorted, counts_sorted, top_n, bottom_n, titulo=titulo)


def relatorio_top_frac_classes(y_base, frac_top_classes=0.20, top_n=10, bottom_n=10, titulo=""):
    """
    1) Seleciona as top (frac_top_classes) CLASSES por frequência em y_base
    2) Dentro desse subset de classes, imprime top-10 e bottom-10 por contagem.

    Ex.: frac_top_classes=0.20 => pega as 20% classes mais frequentes.
    """
    y_base = np.asarray(y_base)

    classes, counts = np.unique(y_base, return_counts=True)
    ordem = np.argsort(-counts)  # desc
    classes_sorted = classes[ordem]
    counts_sorted = counts[ordem]

    n_classes = len(classes_sorted)
    k = int(np.ceil(frac_top_classes * n_classes))
    k = max(1, min(k, n_classes))

    classes_top = classes_sorted[:k]
    mask = np.isin(y_base, classes_top)
    y_sub = y_base[mask]

    print(f"\n[Top {100*frac_top_classes:.1f}% classes] Mantidas {k}/{n_classes} classes | "
          f"Instâncias mantidas: {len(y_sub)}/{len(y_base)} ({100*len(y_sub)/max(1,len(y_base)):.2f}%)")

    # Recalcula contagens dentro do subset de classes
    classes2, counts2 = np.unique(y_sub, return_counts=True)
    ordem2 = np.argsort(-counts2)
    _print_top_bottom_from_counts(classes2[ordem2], counts2[ordem2], top_n, bottom_n, titulo=titulo)

relatorio_top_bottom_classes(y_final, top_n=10, bottom_n=10, titulo="Distribuição no y_final (final_frac)")

if __name__ == "__main__":
    main()
