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
# Utilitários (mantidos no bloco, para colar após seu código)
# ============================================================

def codificar_rotulos(y: np.ndarray):
    """
    Mapeia rótulos arbitrários -> índices 0..K-1.
    Retorna:
      y_idx: (n,)
      classes: (K,) com os rótulos originais na ordem do índice
      mapa: dict rotulo_original -> idx
    """
    classes = np.unique(y)
    mapa = {int(c): i for i, c in enumerate(classes.tolist())}
    y_idx = np.array([mapa[int(v)] for v in y], dtype=np.int64)
    return y_idx, classes.astype(np.int64), mapa

def one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Cria matriz one-hot (n, K) a partir de índices 0..K-1."""
    n = y_idx.shape[0]
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y_idx] = 1.0
    return Y

def calcular_erro_classificacao(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro = fração de amostras incorretas."""
    return float(np.mean(y_true != y_pred))

def mostrar_previsoes_amostrais(y_true: np.ndarray, y_pred: np.ndarray, n_amostras: int = 5, seed: int = 42):
    """Imprime n_amostras previsões aleatórias (verdadeiro -> predito)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(n_amostras, n), replace=False)
    print("\nAmostras aleatórias (verdadeiro -> predito):")
    for t, i in enumerate(idx, start=1):
        print(f"  #{t:02d} | y_true={int(y_true[i])} -> y_pred={int(y_pred[i])}")

# ============================================================
# Gradiente descendente genérico (reutilizável)
# ============================================================

def gradiente_descendente(
    params_iniciais: dict,
    funcao_perda_grad,
    X: np.ndarray,
    Y: np.ndarray,
    taxa_aprendizado: float = 0.05,
    epocas: int = 100,
    tamanho_lote: int | None = 256,
    seed: int = 42,
    verbose: bool = True
):
    """
    Otimizador por gradiente descendente (batch ou mini-batch).

    - params_iniciais: dict de arrays/parâmetros
    - funcao_perda_grad(params, Xb, Yb) -> (perda, grads_dict)
    """
    rng = np.random.default_rng(seed)

    # cópias defensivas
    params = {}
    for k, v in params_iniciais.items():
        params[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else v

    n = X.shape[0]
    if tamanho_lote is None or tamanho_lote <= 0 or tamanho_lote > n:
        tamanho_lote = n

    historico = []

    for epoca in range(epocas):
        idx = rng.permutation(n)

        for ini in range(0, n, tamanho_lote):
            lote = idx[ini:ini + tamanho_lote]
            Xb = X[lote]
            Yb = Y[lote]

            perda, grads = funcao_perda_grad(params, Xb, Yb)

            # atualização
            for nome_param in grads:
                params[nome_param] = params[nome_param] - taxa_aprendizado * grads[nome_param]

        if verbose:
            perda_full, _ = funcao_perda_grad(params, X, Y)
            historico.append(float(perda_full))
            if epocas <= 10 or (epoca % max(1, epocas // 10) == 0) or (epoca == epocas - 1):
                print(f"[GD] Época {epoca+1}/{epocas} | perda={perda_full:.6f}")

    return params, historico

# ============================================================
# Regressão Linear Multiclasse (Least Squares) + Elastic Net
# ============================================================

def perda_e_gradiente_regressao_linear_elasticnet(
    params: dict,
    X: np.ndarray,
    Y: np.ndarray,
    lambda_l1: float = 0.5,
    lambda_l2: float = 0.5
):
    """
    Treina um modelo linear multiclasse via MSE contra targets one-hot.

    params:
      - W: (K, d)
      - b: (K,)

    Predição (scores): S = X @ W.T + b  -> (m, K)

    Loss:
      MSE médio + lambda_l1*||W||_1 + 0.5*lambda_l2*||W||^2
    (regularização NÃO aplica em b)
    """
    W = params["W"]  # (K, d)
    b = params["b"]  # (K,)

    m, d = X.shape
    K = Y.shape[1]
    if W.shape != (K, d):
        raise ValueError(f"W esperado {(K, d)}, recebido {W.shape}")

    S = X @ W.T + b  # (m, K)
    E = S - Y        # (m, K)

    # MSE médio = (1/m) * sum_{i,k} (E^2) / K  (opcional dividir por K p/ escala)
    # Aqui usamos média em todos os elementos:
    mse = np.mean(E * E)

    reg = lambda_l1 * np.sum(np.abs(W)) + 0.5 * lambda_l2 * np.sum(W * W)
    perda_total = mse + reg

    # gradientes do MSE:
    # d/dS mean(E^2) = 2*E / (m*K) se a média for por elementos
    # Como mse = mean(E^2) em todos elementos, o fator é 2/(m*K)
    fator = 2.0 / (m * K)
    dS = fator * E  # (m, K)

    grad_W = dS.T @ X          # (K, d)
    grad_b = np.sum(dS, axis=0)  # (K,)

    # regularização
    grad_W += lambda_l2 * W
    grad_W += lambda_l1 * np.sign(W)

    grads = {"W": grad_W, "b": grad_b}
    return float(perda_total), grads

def treinar_regressao_linear_multiclasse_elasticnet(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    lambda_l1: float = 0.5,
    lambda_l2: float = 0.5,
    taxa_aprendizado: float = 0.05,
    epocas: int = 100,
    tamanho_lote: int | None = 256,
    seed: int = 42,
    verbose: bool = True
):
    """
    Treina regressão linear multiclasse (MSE vs one-hot) com Elastic Net (do zero).
    Retorna um 'modelo' com W, b, e mapeamento de classes.
    """
    y_idx, classes, mapa = codificar_rotulos(y_treino)
    K = len(classes)
    d = X_treino.shape[1]

    Y = one_hot(y_idx, K)  # (n, K)

    rng = np.random.default_rng(seed)
    W0 = rng.normal(loc=0.0, scale=0.01, size=(K, d)).astype(np.float64)
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
        params0,
        f,
        X_treino.astype(np.float64, copy=False),
        Y,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        tamanho_lote=tamanho_lote,
        seed=seed,
        verbose=verbose
    )

    modelo = {
        "W": params_finais["W"],
        "b": params_finais["b"],
        "classes": classes,
        "mapa": mapa,
        "historico_perda": historico
    }
    return modelo

def prever_regressao_linear_multiclasse(modelo: dict, X: np.ndarray):
    """
    Prediz classe pelo argmax dos scores lineares.
    Retorna y_pred (rótulos originais) e scores (m, K).
    """
    W = modelo["W"]
    b = modelo["b"]
    classes = modelo["classes"]

    scores = X.astype(np.float64, copy=False) @ W.T + b  # (m, K)
    idx_pred = np.argmax(scores, axis=1)
    y_pred = classes[idx_pred]
    return y_pred, scores

def treinar_e_avaliar_regressao_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_l1: float = 0.5,
    lambda_l2: float = 0.5,
    taxa_aprendizado: float = 0.05,
    epocas: int = 100,
    tamanho_lote: int | None = 256,
    seed: int = 42,
    verbose: bool = True
):
    """
    Treina no treino, avalia no teste, imprime erro e 5 previsões amostrais.
    Retorna (modelo, y_pred_test).
    """
    # aviso se existirem classes no teste fora do treino
    classes_treino = set(np.unique(y_train).tolist())
    classes_teste = set(np.unique(y_test).tolist())
    fora = classes_teste - classes_treino
    if len(fora) > 0:
        print(f"[Aviso] Existem {len(fora)} classes no teste que não aparecem no treino. "
              "Isso tende a aumentar o erro (inevitável).")

    modelo = treinar_regressao_linear_multiclasse_elasticnet(
        X_train, y_train,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        tamanho_lote=tamanho_lote,
        seed=seed,
        verbose=verbose
    )

    y_pred_test, _ = prever_regressao_linear_multiclasse(modelo, X_test)

    erro = calcular_erro_classificacao(y_test.astype(np.int64), y_pred_test.astype(np.int64))
    print(f"\nErro no teste: {erro:.4f} | Acurácia: {1.0 - erro:.4f}")

    mostrar_previsoes_amostrais(y_test.astype(np.int64), y_pred_test.astype(np.int64), n_amostras=5, seed=seed)
    return modelo, y_pred_test

# ============================================================
# Exemplo de uso (rodar após split)
# ============================================================

modelo_ls, y_pred_test = treinar_e_avaliar_regressao_linear(
     X_train, y_train,
     X_test, y_test,
     lambda_l1=0.5, lambda_l2=0.5,
     taxa_aprendizado=0.05,
     epocas=100,
     tamanho_lote=256,
     seed=42,
     verbose=True
 )

