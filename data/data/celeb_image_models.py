import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from itertools import combinations

#Antes de rodar lembrar de rodar o HOG e definir qual a % dos dados!
#Para experimentação usar HOG com 30% dos dados

# Configurando diretórios
DATA_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / "celeba_hog_128x128_o9.joblib"

print("Dataset path:", DATASET_PATH)
print("Exists?", DATASET_PATH.exists())

#Demora 30s-60s
data = joblib.load(DATASET_PATH)
X, y = data["X"], data["y"]

#Checagem das classes - somente conferência. Encontra 10177 classes
print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("n classes:", len(np.unique(y)))

# ============================
# Split: 90% train, 1% val, 9% test
# ============================
TEST_FRAC_FULL = 0.09
VAL_FRAC_FULL  = 0.01
TRAIN_FRAC_FULL = 0.90

# 1) Test: 9% do total
X_rest, X_test, y_rest, y_test = train_test_split(
    X,
    y,
    test_size=TEST_FRAC_FULL,
    random_state=42,
    shuffle=True
    # sem stratify (mantendo seu padrão)
)

# 2) Val: 1% do total, tirado dos 91% restantes
#    fração dentro do restante = 0.00 / 0.91
val_frac_rest = VAL_FRAC_FULL / (1.0 - TEST_FRAC_FULL)

X_train, X_val, y_train, y_val = train_test_split(
    X_rest,
    y_rest,
    test_size=val_frac_rest,
    random_state=42,
    shuffle=True
    # sem stratify
)

# opcional: liberar memória
del X, y, X_rest, y_rest

print("Train:", X_train.shape, y_train.shape)
print("Val:",   X_val.shape,   y_val.shape)
print("Test:",  X_test.shape,  y_test.shape)
print("Total checado:", len(y_train) + len(y_val) + len(y_test))



# ============================================================
# Utilitários
# ============================================================

def sigmoide(z: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estável."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))

def ajustar_padronizador(X_treino: np.ndarray, eps: float = 1e-8):
    """Calcula média e desvio padrão por feature para padronização."""
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    desvio = np.where(desvio < eps, 1.0, desvio)
    return media, desvio

def aplicar_padronizador(X: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    """Aplica padronização (z-score)."""
    return (X - media) / desvio

def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_validas_set: set):
    """Mantém apenas amostras cujos rótulos pertencem ao conjunto de classes."""
    mascara = np.isin(y, np.array(list(classes_validas_set)))
    return X[mascara], y[mascara]

# ============================================================
# Função de perda + gradiente (Logística Binária + Elastic Net)
# ============================================================

def perda_e_gradiente_logistico_elasticnet(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    lambda_l1: float = 0.5,
    lambda_l2: float = 0.5
):
    """
    Retorna (perda, grads) para regressão logística binária com Elastic Net.
    y deve ser {0,1}. Regularização NÃO aplica no bias.
    Perda = média( log(1+exp(z)) - y*z ) + lambda_l1*||w||_1 + 0.5*lambda_l2*||w||^2
    """
    w = params["w"]
    b = params["b"]

    m = X.shape[0]
    z = X @ w + b

    # Cross-entropy estável: log(1+exp(z)) - y*z  == logaddexp(0,z) - y*z
    perda_dados = np.mean(np.logaddexp(0.0, z) - y * z)

    # Elastic Net (L1 + L2)
    perda_reg = lambda_l1 * np.sum(np.abs(w)) + 0.5 * lambda_l2 * np.sum(w * w)

    perda_total = perda_dados + perda_reg

    p = sigmoide(z)
    erro = (p - y)  # shape (m,)

    grad_w = (X.T @ erro) / m
    grad_b = float(np.mean(erro))

    # grad regularização
    grad_w += lambda_l2 * w
    grad_w += lambda_l1 * np.sign(w)  # subgradiente do L1; sign(0)=0

    grads = {"w": grad_w, "b": grad_b}
    return perda_total, grads

# ============================================================
# Gradiente Descendente Genérico (reutilizável)
# ============================================================

def gradiente_descendente(
    params_iniciais: dict,
    funcao_perda_grad,
    X: np.ndarray,
    y: np.ndarray,
    taxa_aprendizado: float = 0.1,
    epocas: int = 200,
    tamanho_lote: int | None = None,
    seed: int = 42,
    verbose: bool = False
):
    """
    Otimizador genérico por gradiente descendente (batch ou mini-batch).
    - funcao_perda_grad(params, X_batch, y_batch) -> (perda, grads_dict)
    Retorna (params_finais, historico_perda).
    """
    rng = np.random.default_rng(seed)
    params = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in params_iniciais.items()}
    historico = []

    n = X.shape[0]
    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n:
        tamanho_lote = n  # batch total

    for epoca in range(epocas):
        idx = rng.permutation(n)

        for ini in range(0, n, tamanho_lote):
            lote = idx[ini:ini + tamanho_lote]
            Xb = X[lote]
            yb = y[lote]

            perda, grads = funcao_perda_grad(params, Xb, yb)

            # atualização
            params["w"] = params["w"] - taxa_aprendizado * grads["w"]
            params["b"] = params["b"] - taxa_aprendizado * grads["b"]

        if verbose and (epoca % max(1, epocas // 10) == 0 or epoca == epocas - 1):
            perda_full, _ = funcao_perda_grad(params, X, y)
            historico.append(float(perda_full))
            print(f"[GD] Época {epoca+1}/{epocas} | perda={perda_full:.6f}")

    return params, historico

# ============================================================
# Treino/Predict Logístico Binário (wrapper)
# ============================================================

def treinar_logistico_binario_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    lambda_l1: float = 0.5,
    lambda_l2: float = 0.5,
    taxa_aprendizado: float = 0.1,
    epocas: int = 200,
    tamanho_lote: int | None = 256,
    seed: int = 42,
    verbose: bool = False
):
    """Treina um classificador binário logístico (do zero) e retorna params."""
    n_features = X.shape[1]
    params0 = {"w": np.zeros(n_features, dtype=np.float64), "b": 0.0}

    def f(params, Xb, yb):
        return perda_e_gradiente_logistico_elasticnet(
            params, Xb, yb, lambda_l1=lambda_l1, lambda_l2=lambda_l2
        )

    params_finais, historico = gradiente_descendente(
        params0, f, X, y,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        tamanho_lote=tamanho_lote,
        seed=seed,
        verbose=verbose
    )
    return params_finais, historico

def prever_proba_binario(params: dict, X: np.ndarray) -> np.ndarray:
    """Retorna P(y=1|x) para o modelo binário."""
    return sigmoide(X @ params["w"] + params["b"])

# ============================================================
# OvO (One-vs-One) Multiclasse
# ============================================================

def selecionar_classes_mais_frequentes(y: np.ndarray, max_classes: int, seed: int = 42) -> np.ndarray:
    """
    Seleciona as max_classes classes mais frequentes (determinístico).
    """
    classes, contagens = np.unique(y, return_counts=True)
    ordem = np.argsort(-contagens)  # decrescente
    selecionadas = classes[ordem[:max_classes]]
    return selecionadas

def treinar_ovo_logistico_elasticnet(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    lambda_l1: float = 0.5,
    lambda_l2: float = 0.5,
    taxa_aprendizado: float = 0.1,
    epocas: int = 200,
    tamanho_lote: int | None = 256,
    max_classes: int | None = 20,
    padronizar: bool = True,
    seed: int = 42,
    verbose: bool = False
):
    """
    Treina um conjunto OvO de regressões logísticas binárias.
    Retorna um 'modelo' (dict) com:
      - classes_modelo (np.array)
      - mapa_classes (dict label->idx)
      - padronizador (media, desvio) ou None
      - classificadores: lista de dicts {par:(ci,cj), params:...}
    """
    rng = np.random.default_rng(seed)

    # 1) escolher classes (evitar explosão combinatória)
    classes_unicas = np.unique(y_treino)
    if max_classes is not None:
        if max_classes < 2:
            raise ValueError("max_classes precisa ser >= 2 para OvO.")
        classes_modelo = selecionar_classes_mais_frequentes(y_treino, max_classes=max_classes, seed=seed)
    else:
        classes_modelo = classes_unicas

    classes_modelo = np.array(classes_modelo)
    classes_set = set(classes_modelo.tolist())

    # 2) filtrar treino para apenas essas classes
    Xf, yf = filtrar_por_classes(X_treino, y_treino, classes_set)

    # 3) padronização opcional
    padronizador = None
    if padronizar:
        media, desvio = ajustar_padronizador(Xf)
        Xf = aplicar_padronizador(Xf, media, desvio)
        padronizador = (media, desvio)

    mapa_classes = {int(c): i for i, c in enumerate(classes_modelo.tolist())}

    # 4) treinar todos os pares
    pares = list(combinations(classes_modelo.tolist(), 2))
    if verbose:
        print(f"[OvO] Classes no modelo: {len(classes_modelo)} | Pares OvO: {len(pares)}")

    classificadores = []
    for idx_par, (ci, cj) in enumerate(pares, start=1):
        # filtra amostras do par
        mascara = (yf == ci) | (yf == cj)
        X_par = Xf[mascara]
        y_par = yf[mascara]

        # y binário: 1 para ci, 0 para cj
        y_bin = (y_par == ci).astype(np.float64)

        params, _ = treinar_logistico_binario_elasticnet(
            X_par, y_bin,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            taxa_aprendizado=taxa_aprendizado,
            epocas=epocas,
            tamanho_lote=tamanho_lote,
            seed=seed,
            verbose=False
        )
        classificadores.append({"par": (ci, cj), "params": params})

        if verbose and (idx_par % max(1, len(pares)//10) == 0 or idx_par == len(pares)):
            print(f"[OvO] Treinado {idx_par}/{len(pares)} pares")

    modelo = {
        "classes_modelo": classes_modelo,
        "mapa_classes": mapa_classes,
        "padronizador": padronizador,
        "classificadores": classificadores
    }
    return modelo

def prever_ovo(modelo: dict, X: np.ndarray):
    """
    Prediz classe por OvO usando 'soma de evidências' (reduz empates):
      - para cada par (ci,cj): soma p em ci e (1-p) em cj
    Retorna (y_pred, scores).
    """
    classes_modelo = modelo["classes_modelo"]
    mapa = modelo["mapa_classes"]
    padronizador = modelo["padronizador"]

    Xp = X
    if padronizador is not None:
        media, desvio = padronizador
        Xp = aplicar_padronizador(Xp, media, desvio)

    n = Xp.shape[0]
    k = len(classes_modelo)
    scores = np.zeros((n, k), dtype=np.float32)

    for clf in modelo["classificadores"]:
        ci, cj = clf["par"]
        i = mapa[int(ci)]
        j = mapa[int(cj)]

        p = prever_proba_binario(clf["params"], Xp).astype(np.float32)
        scores[:, i] += p
        scores[:, j] += (1.0 - p)

    idx_pred = np.argmax(scores, axis=1)
    y_pred = classes_modelo[idx_pred]
    return y_pred, scores

# ============================================================
# Avaliação + Amostras
# ============================================================

def calcular_erro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro de classificação (1 - acurácia)."""
    return float(np.mean(y_true != y_pred))

def mostrar_previsoes_amostrais(y_true: np.ndarray, y_pred: np.ndarray, n_amostras: int = 5, seed: int = 42):
    """Mostra n_amostras previsões aleatórias (verdadeiro vs predito)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(n_amostras, n), replace=False)

    print("\nAmostras aleatórias (verdadeiro -> predito):")
    for t, i in enumerate(idx, start=1):
        print(f"  #{t:02d} | y_true={int(y_true[i])} -> y_pred={int(y_pred[i])}")

# ============================================================
# Exemplo de uso (rode depois do seu split)
# ============================================================

# IMPORTANTE: para OvO não explodir, comece com max_classes pequeno (ex.: 10~30).
modelo_ovo = treinar_ovo_logistico_elasticnet(
    X_train, y_train,
    lambda_l1=0.5, lambda_l2=0.5,
    taxa_aprendizado=0.1,
    epocas=200,
    tamanho_lote=256,
    max_classes=20,       # <-- ajuste aqui (None = todas as classes; geralmente inviável)
    padronizar=True,
    seed=42,
    verbose=True
)

# Filtra teste para apenas as classes presentes no modelo
classes_set = set(modelo_ovo["classes_modelo"].tolist())
X_test_f, y_test_f = filtrar_por_classes(X_test, y_test, classes_set)

y_pred, _ = prever_ovo(modelo_ovo, X_test_f)
erro = calcular_erro(y_test_f, y_pred)

print(f"\nErro no teste (classes do modelo): {erro:.4f} | Acurácia: {1-erro:.4f}")
mostrar_previsoes_amostrais(y_test_f, y_pred, n_amostras=5, seed=42)
