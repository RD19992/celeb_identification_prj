# -*- coding: utf-8 -*-
"""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Treinamento de CNN tradicional com K-Fold CV

Arquivos gerados (por execução), dentro de:
  <DATASET_DIR>/runs/<YYYYMMDD_HHMMSS>/

- run_errors.csv            (métricas por fold/época)
- summary.json              (agregados e melhores erros por fold)
- hyperparameters.json      (CONFIG usado na execução)
- label2idx.json            (mapeamento label_original -> y_reindexado)
- [opcional] models/*.keras (salvo apenas se SAVE_MODEL=True)



------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import csv
import datetime as _dt
import json
import math
import random
import time
from typing import Dict, Tuple, List, Any, Optional, Set
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras import mixed_precision

# ---------------------------------------------------------------------------
# Referências
# ---------------------------------------------------------------------------
# Este script é uma implementação prática de um pipeline de treino/validação
# de CNNs em Keras/TensorFlow.
# Referências-base (citadas nos comentários):
#  - [LeCun98] CNNs e conv/pooling na prática (LeCun et al., 1998)
#  - [He15] ReLU/rectifiers e inicialização 'He' (He et al., 2015)
#  - [Ioffe15] Batch Normalization (Ioffe & Szegedy, 2015)
#  - [Srivastava14] Dropout (Srivastava et al., 2014)
#  - [Krogh91] / [KroghHertz92] Weight decay / L2 como regularização
#  - [KingmaBa14] Adam (Kingma & Ba, 2014)
#  - [LoshchilovHutter16] Cosine annealing + warm restarts (SGDR)
#  - [Prechelt97] Early stopping (critério via validação)
#  - [Stone74] / [Kohavi95] (Stratified) K-fold cross-validation
#  - [Russakovsky15] ImageNet (contexto para mean/std e benchmarks)
#  - [Shorten19] Data augmentation (survey)
#  - [Micikevicius17] Mixed precision training (FP16/FP32 + loss scaling)


# ============================================================
# (1) SELEÇÃO DE ADAPTADOR / BACKEND (DirectML-friendly)
# ============================================================

os.environ["DML_VISIBLE_DEVICES"] = "0"  # mantém apenas o adapter 0 (GPU discreta)


mixed_precision.set_global_policy("float32")

# >>> Logar onde cada op é colocado (CPU/GPU)
tf.debugging.set_log_device_placement(False)


# ============================================================
# CONFIGURAÇÕES
# ============================================================




# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    # --------------------------------------------------------
    # Dados produzidos por script de ingestão
    # --------------------------------------------------------
    "DATASET_DIR": Path(__file__).resolve().parent / "celeba_rgb_128x128",
    "MANIFEST_NAME": "manifest.csv",
    "ONLY_OK": True,                 # usa ok==True se a coluna existir

    # --------------------------------------------------------
    # Filtragem de classes
    # --------------------------------------------------------
    "TOP_CLASS_FRACTION": 0.01,       # top fração de classes mais frequentes
    "KFOLDS": 5,
    "SEED": 42,

    # --------------------------------------------------------
    # Split de TESTE FINAL (hold-out ANTES do CV)
    # --------------------------------------------------------
    # Fração de classes (identidades) que participarão do teste final.
    # 1.0 => todas as classes elegíveis terão um pedaço reservado para teste.
    # 0.3 => ~30% das classes (amostradas aleatoriamente) terão imagens reservadas para teste.
    "FINAL_TEST_CLASS_FRACTION": 1.0,

    # Para cada classe escolhida acima, reservamos esta fração de imagens para teste final.
    # Importante: garantimos que sobrem pelo menos KFOLDS imagens no conjunto do CV para essa classe.
    "FINAL_TEST_FRACTION_PER_CLASS": 0.20,

    # Para o treino final (antes do teste), separamos uma validação interna
    # (para early stopping do treino final), por classe.
    "FINAL_TRAIN_VAL_FRACTION_PER_CLASS": 0.20,

    # --------------------------------------------------------
    # Entrada
    # --------------------------------------------------------

    # [LeCun98] define a motivação para operar em grades 2D (imagens) com
    # convoluções e pooling; fixar resolução (IMG_SIZE) e canais.
    # Normalização por média/desvio padrão é uma forma de padronização do input;
    # os valores abaixo são o padrão do ImageNet (muito usado em CV), cf. [Russakovsky15].
    "IMG_SIZE": 128,
    "IN_CHANNELS": 3,
    "NORM_MEAN": (0.485, 0.456, 0.406),
    "NORM_STD":  (0.229, 0.224, 0.225),

    # --------------------------------------------------------
    # Aumentação de dados
    # --------------------------------------------------------

    # Aumentação (flip/crop/jitter etc.) age como regularização “no espaço dos dados”,
    # melhorando generalização quando o dataset efetivo é limitado ou enviesado.
    # Ver síntese em [Shorten19].
    "AUG_HFLIP": True,
    "AUG_PAD": 4,                    # reflect-pad + random crop; 0 desabilita

    # --------------------------------------------------------
    # --------------------------------------------------------
    # CNN tradicional
    # --------------------------------------------------------
    # Camadas: Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense -> Dense(softmax)
    "CNN_CONV1_FILTERS": 32,
    "CNN_CONV1_KERNEL": (3, 3),
    "CNN_CONV1_STRIDES": (1, 1),
    "CNN_CONV1_PADDING": "valid",
    "CNN_CONV1_ACTIVATION": "relu",

    "CNN_POOL1_SIZE": (2, 2),
    "CNN_POOL1_STRIDES": (2, 2),

    "CNN_CONV2_FILTERS": 64,
    "CNN_CONV2_KERNEL": (3, 3),
    "CNN_CONV2_STRIDES": (1, 1),
    "CNN_CONV2_PADDING": "valid",
    "CNN_CONV2_ACTIVATION": "relu",

    "CNN_POOL2_SIZE": (2, 2),
    "CNN_POOL2_STRIDES": (2, 2),

    "CNN_DENSE_UNITS": 64,
    "CNN_DENSE_ACTIVATION": "relu",

    # Dropout opcional na "head"
    "CNN_HEAD_DROPOUT": 0.3,

    # Saída softmax
    "CNN_OUTPUT_ACTIVATION": "softmax",

    # Perda (categorical cross-entropy).
    # Em problemas com MUITAS classes, SparseCategoricalCrossentropy é equivalente e costuma ser mais eficiente.
    "USE_SPARSE_CE": True,

    # --------------------------------------------------------
    # Regularização
    # --------------------------------------------------------

    # Regularização:
    #  - Dropout: desativa unidades aleatoriamente durante o treino para reduzir coadaptação
    #    e overfitting [Srivastava14].
    #  - L2/weight decay: penaliza pesos grandes;
    #    [Krogh91] / [KroghHertz92].
    "L2_WEIGHT": 1e-4,                # L2 no kernel de Conv/Dense

    # --------------------------------------------------------
    # Treino
    # --------------------------------------------------------
    "BATCH_SIZE": 32,

    "EPOCHS": 50,                     # - serve como default

    "EPOCHS_CV": 10,                  # max épocas por fold no CV
    "EPOCHS_FINAL": 40,               # max épocas do treino final (antes do teste final)

    # LR base, usado como "initial_lr" do schedule (se habilitado).
    "LR": 3e-4,

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    # Early stopping interrompe o treino quando a métrica de validação para de melhorar,
    # reduzindo overfitting e economizando compute. Critérios/prática em [Prechelt97].
    # Desligar, colocar 0.
    "EARLY_STOPPING_PATIENCE_CV": 5,
    "EARLY_STOPPING_PATIENCE_FINAL": 10,
    "EARLY_STOPPING_MIN_DELTA": 0.0001,   # melhora mínima exigida (em termos de val_err)

    # --------------------------------------------------------
    # Learning-rate schedule (Adam)
    # --------------------------------------------------------

    # Schedules de LR (decay/cosine/warm restarts) controlam o “tamanho do passo” ao longo
    # do treino e costumam ser uma das alavancas mais fortes para convergência e generalização.
    # Cosine annealing + warm restarts popularizou-se com SGDR [LoshchilovHutter16].
    # TYPE:
    # - "constant"
    # - "exponential_decay"
    # - "cosine_decay"
    # - "cosine_decay_restarts"
    "LR_SCHEDULE_TYPE": "cosine_decay",

    # Parâmetros p/ exponential_decay
    "LR_DECAY_EPOCHS": 2,          # a cada quantas épocas (em passos) aplica decaimento
    "LR_DECAY_RATE": 0.95,
    "LR_STAIRCASE": False,

    # Parâmetros p/ cosine_decay / cosine_decay_restarts
    "LR_COSINE_ALPHA": 0.0,           # LR final = alpha * initial_lr (no limite)
    "LR_COSINE_FIRST_DECAY_EPOCHS": 2.0,  # só para restarts
    "LR_COSINE_T_MUL": 2.0,           # só para restarts
    "LR_COSINE_M_MUL": 1.0,           # só para restarts

    # --------------------------------------------------------
    # Performance / tf.data
    # --------------------------------------------------------
    "PREFETCH": True,
    "CACHE_DATASET": False,           # True só se fizer sentido em RAM/SSD
    "DETERMINISTIC": False,           # False = mais rápido (ordem não garantida)

    # Se a ingestão já gerou JPEGs IMG_SIZE x IMG_SIZE, pula resize.
    # Se False, faz resize em runtime (mais lento porém robusto).
    "ASSUME_INGESTED_SIZE": True,

    # Método DCT do decoder JPEG (pode influenciar velocidade/precisão)
    "JPEG_DCT_METHOD": "INTEGER_FAST",  # "INTEGER_FAST" ou "INTEGER_ACCURATE"

    # Mixed precision / XLA
    "MIXED_PRECISION": False,
    "XLA": False,                      # jit_compile=True nos passos
    "ALLOW_TF32": False,               # CUDA Ampere+: acelera matmul com TF32

    # Debug / contadores
    "LOG_EVERY_N_STEPS": 50,
    "PRINT_FIRST_BATCH_INFO": True,

    # Matriz de confusão no conjunto de teste
    "SAVE_CONFUSION_MATRIX": True,
    # Segurança: evita explodir RAM/CSV em problemas com muitas classes
    "CONFUSION_MATRIX_MAX_CLASSES": 500,

    # Confirmação de resolução (lê header JPEG; sem resize)
    "RESOLUTION_SAMPLE_N": 256,
    "STRICT_RESOLUTION_CHECK": False,  # True => falha se achar imagem fora do alvo

    # Saídas
    "RUNS_DIRNAME": "runs",
    "SAVE_MODEL": False,              # começa desabilitado (mantido) - salva modelos por fold no CV

    # NOVO: salvar modelo final + pesos txt ao final
    "SAVE_FINAL_MODEL": True,
    "FINAL_MODEL_FILENAME": "final_model_best.keras",
    "FINAL_WEIGHTS_TXT_FILENAME": "final_model_best_weights.txt",

    # --------------------------------------------------------
    # (2) FORÇAR GPU / CPU (opcional)
    # --------------------------------------------------------
    # Se FORCE_GPU=True, tentamos travar o runtime para usar /GPU:0.
    # Se FORCE_GPU_STRICT=True e não houver GPU, levantamos erro.
    "FORCE_GPU": False,
    "FORCE_GPU_STRICT": False,

    # Opcional: forçar CPU (útil para debug/comparação)
    "FORCE_CPU": False,

    # Opcional: diagnóstico (não altera lógica; só imprime)
    "PRINT_DEVICE_DIAGNOSTICS": True,
}


# ============================================================
# GPU/CPU SETUP
# ============================================================
def _print_visible_devices() -> None:
    """Imprime dispositivos visíveis (diagnóstico)."""
    try:
        vis = tf.config.get_visible_devices()
        print("[DBG] Visible devices:")
        for d in vis:
            print("   -", d)
    except Exception as e:
        print("[DBG] Could not read visible devices:", e)

    try:
        phys_cpu = tf.config.list_physical_devices("CPU")
        phys_gpu = tf.config.list_physical_devices("GPU")
        print(f"[DBG] Physical CPU count: {len(phys_cpu)} | Physical GPU count: {len(phys_gpu)}")
        if phys_gpu:
            for i, g in enumerate(phys_gpu):
                print(f"      GPU[{i}]: {g}")
    except Exception as e:
        print("[DBG] Could not list physical devices:", e)


def configure_devices(cfg: Dict[str, Any]) -> str:
    """
    Decide e configura o 'device' principal (/GPU:0 ou /CPU:0).

    Ideia didática:
    - O TensorFlow pode enxergar GPU mas, se certos kernels não existirem
      (ou se o backend estiver mal instalado), ele faz fallback para CPU.

    Importante:
    - Mesmo forçando /GPU:0, algumas ops podem cair na CPU se não houver kernel GPU.
    """
    force_cpu = bool(cfg.get("FORCE_CPU", False))
    force_gpu = bool(cfg.get("FORCE_GPU", False))
    strict_gpu = bool(cfg.get("FORCE_GPU_STRICT", False))

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    if force_cpu:
        try:
            # deixa só CPU visível
            tf.config.set_visible_devices(cpus, "CPU")
            # esconde GPU
            if gpus:
                tf.config.set_visible_devices([], "GPU")
            print("[INFO] FORCE_CPU=True -> usando CPU.")
        except Exception as e:
            print("[WARN] Could not force CPU visibility:", e)
        return "/CPU:0"

    if force_gpu:
        if not gpus:
            msg = "[WARN] FORCE_GPU=True mas nenhuma GPU foi detectada."
            if strict_gpu:
                raise RuntimeError(msg + " (FORCE_GPU_STRICT=True)")
            print(msg + " Fazendo fallback para CPU.")
            return "/CPU:0"

        try:
            # usa só a primeira GPU (comum em setups DirectML/CUDA)
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("[INFO] FORCE_GPU=True -> usando GPU:", gpus[0])
        except Exception as e:
            msg = f"[WARN] Could not set GPU config: {e}"
            if strict_gpu:
                raise RuntimeError(msg)
            print(msg, " Fazendo fallback para CPU.")
            return "/CPU:0"

        return "/GPU:0"

    # Comportamento padrão: usa GPU se existir, senão CPU.
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("[INFO] Using GPU:", gpus[0])
        except Exception as e:
            print("[WARN] Could not set GPU config:", e)
            print("[INFO] Falling back to CPU.")
            return "/CPU:0"
        return "/GPU:0"

    print("[INFO] No GPU found, using CPU.")
    return "/CPU:0"


# Configura e grava o device escolhido no CONFIG
CONFIG["DEVICE"] = configure_devices(CONFIG)

if bool(CONFIG.get("PRINT_DEVICE_DIAGNOSTICS", True)):
    _print_visible_devices()


# =========================
# UTIL
# =========================
def set_seed(seed: int) -> None:
    """
    Define seeds para reprodutibilidade.
    Didática:
    - random: Python puro
    - numpy: operações vetoriais/embaralhamento
    - tf: ops internas (ex.: augment, init de pesos)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def now_stamp() -> str:
    """Timestamp usado para criar pasta de runs."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    """Cria diretório (incluindo pais) se não existir."""
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    """Salva JSON garantindo que Path seja serializável."""
    def _default(x):
        if isinstance(x, Path):
            return str(x)
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def setup_tf_performance(cfg: Dict[str, Any]) -> None:
    """
    Ajustes de performance opcionais:
    - TF32 (NVIDIA Ampere+): acelera matmul com precisão próxima de FP32.
    - Mixed precision: usa float16 em partes do grafo (normalmente acelera em GPU).
    """
    if bool(cfg.get("ALLOW_TF32", False)):
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("[INFO] TF32 enabled (if supported by your GPU backend).")
        except Exception:
            pass

    if bool(cfg.get("MIXED_PRECISION", False)):
        try:
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled: mixed_float16")
        except Exception as e:
            print("[WARN] Could not enable mixed precision:", e)


# =========================
# RESOLUTION CONFIRMATION
# =========================
def confirm_image_resolution(df: pd.DataFrame, cfg: Dict[str, Any], sample_n: Optional[int] = None) -> None:
    """
    Confirma resolução ingerida (antes de qualquer resize) por amostragem.
    Importante:
    - tf.io.extract_jpeg_shape lê só o header do JPEG (rápido), não decodifica a imagem inteira.
    - Isso checa se sua ingestão realmente produziu IMG_SIZE x IMG_SIZE.

    Se STRICT_RESOLUTION_CHECK=True:
    - o script falha caso encontre alguma imagem fora do tamanho.
    """
    n = int(sample_n or cfg.get("RESOLUTION_SAMPLE_N", 256))
    n = min(n, len(df))
    if n <= 0:
        print("[WARN] Resolution check skipped (empty df).")
        return

    target = (int(cfg["IMG_SIZE"]), int(cfg["IMG_SIZE"]))
    sample = df.sample(n=n, random_state=int(cfg["SEED"]))["dst"].astype(str).tolist()

    counts: Dict[Tuple[int, int, int], int] = {}
    bad: List[Tuple[str, Tuple[int, int, int]]] = []

    for p in sample:
        raw = tf.io.read_file(p)
        shp = tf.io.extract_jpeg_shape(raw)  # [H,W,C]
        h, w, c = int(shp[0].numpy()), int(shp[1].numpy()), int(shp[2].numpy())
        key = (h, w, c)
        counts[key] = counts.get(key, 0) + 1
        if (h, w) != target:
            bad.append((p, key))

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("[INFO] Ingested resolution check (sample):")
    for (h, w, c), cnt in top:
        print(f"  - {h}x{w}x{c}: {cnt}/{n}")
    if len(counts) > 10:
        print(f"  ... ({len(counts)} unique resolutions in sample)")

    if bad:
        print(f"[WARN] Found {len(bad)}/{n} images not equal to target {target}. Examples:")
        for p, (h, w, c) in bad[:5]:
            print(f"  - {p} -> {h}x{w}x{c}")
        if bool(cfg.get("STRICT_RESOLUTION_CHECK", False)):
            raise ValueError(
                f"STRICT_RESOLUTION_CHECK enabled: found sampled images not {target}. "
                "Set STRICT_RESOLUTION_CHECK=False or fix ingestion."
            )
    else:
        print(f"[INFO] All sampled images match target resolution {target} ✅")


# =========================
# SPLITS ESTRATIFICADOS (por classe) PARA HOLD-OUT
# =========================
def _pick_holdout_classes(
    ys: List[int],
    class_fraction: float,
    seed: int,
    seed_offset: int = 0,
) -> Set[int]:
    """
    Seleciona um subconjunto de classes para terem amostras reservadas (hold-out).
    - class_fraction=1.0 => todas as classes
    - class_fraction=0.0 => nenhuma classe
    """
    ys = sorted(list(ys))
    if class_fraction <= 0.0:
        return set()
    if class_fraction >= 1.0:
        return set(ys)

    n = int(round(class_fraction * len(ys)))
    n = max(1, min(n, len(ys)))
    rng = np.random.default_rng(int(seed) + int(seed_offset))
    chosen = rng.choice(np.array(ys, dtype=np.int64), size=n, replace=False).tolist()
    return set(int(x) for x in chosen)


def stratified_holdout_split_by_class(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    holdout_class_fraction: float,
    holdout_fraction_per_class: float,
    min_keep_per_class: int,
    seed_offset: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Cria um hold-out (teste/val) ANTES do CV, escolhendo:
    - uma fração das classes (holdout_class_fraction)
    - uma fração das imagens dentro de cada classe escolhida (holdout_fraction_per_class)

    Importante:
    - garantimos que o "resto" (train/cv) mantém pelo menos min_keep_per_class exemplos por classe,
      senão não tiramos nada dessa classe.
    """
    if holdout_fraction_per_class <= 0.0 or holdout_class_fraction <= 0.0:
        # Sem holdout (retorna df inteiro como "train" e holdout vazio)
        return df.reset_index(drop=True), df.iloc[0:0].copy(), []

    ys = df["y"].unique().tolist()
    holdout_classes = _pick_holdout_classes(
        ys=ys,
        class_fraction=float(holdout_class_fraction),
        seed=int(cfg["SEED"]),
        seed_offset=int(seed_offset),
    )

    holdout_indices: List[int] = []

    for y in sorted(list(holdout_classes)):
        grp_idx = df.index[df["y"] == int(y)].to_numpy(dtype=np.int64)
        if len(grp_idx) <= min_keep_per_class:
            continue

        # RNG por classe para manter reproduzibilidade
        rng = np.random.default_rng(int(cfg["SEED"]) + int(seed_offset) + 10007 * int(y))
        rng.shuffle(grp_idx)

        n_total = int(len(grp_idx))
        desired = int(round(float(holdout_fraction_per_class) * n_total))

        # não podemos tirar mais do que (n_total - min_keep_per_class)
        max_allowed = max(0, n_total - int(min_keep_per_class))

        n_hold = min(max_allowed, desired)

        # Se a pessoa pediu fração > 0 mas o arredondamento deu 0,
        # tentamos tirar 1 (desde que não viole o min_keep_per_class).
        if desired == 0 and holdout_fraction_per_class > 0.0 and max_allowed > 0:
            n_hold = 1

        if n_hold > 0:
            holdout_indices.extend(grp_idx[:n_hold].tolist())

    holdout_set = set(int(i) for i in holdout_indices)
    holdout_df = df.loc[sorted(list(holdout_set))].copy()
    train_df = df.drop(index=sorted(list(holdout_set))).copy()

    holdout_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    return train_df, holdout_df, sorted(list(holdout_classes))


# =========================
# CNN TRADICIONAL
# =========================
def _as_tuple2(v, name: str) -> Tuple[int, int]:
    """Aceita int, lista/tupla de 2 ints, ou string tipo "3,3" e devolve (a,b)."""
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    if isinstance(v, int):
        return int(v), int(v)
    if isinstance(v, str):
        parts = [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
        if len(parts) == 1:
            k = int(parts[0])
            return k, k
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Config inválida para {name}: {v!r} (esperado int ou par de ints)")


def build_cnn(cfg: Dict[str, Any], num_classes: int) -> Model:
    """
    CNN tradicional:
      Conv2D(32, (3,3), relu) -> MaxPool(2,2) ->
      Conv2D(64, (3,3), relu) -> MaxPool(2,2) ->
      Flatten -> Dense(128, relu) -> Dense(num_classes, softmax)

    Todos os parâmetros são configuráveis via CONFIG (prefixo CNN_*).
    Regularização L2 segue cfg["L2_WEIGHT"] (como no seu frame).
    """
    img_size = int(cfg["IMG_SIZE"])
    in_ch = int(cfg["IN_CHANNELS"])

    l2_w = float(cfg.get("L2_WEIGHT", 0.0))
    kreg = (regularizers.l2(l2_w) if l2_w and l2_w > 0 else None)

    conv1_filters = int(cfg["CNN_CONV1_FILTERS"])
    conv1_kernel = _as_tuple2(cfg["CNN_CONV1_KERNEL"], "CNN_CONV1_KERNEL")
    conv1_strides = _as_tuple2(cfg["CNN_CONV1_STRIDES"], "CNN_CONV1_STRIDES")
    conv1_padding = str(cfg["CNN_CONV1_PADDING"]).lower()
    conv1_act = str(cfg["CNN_CONV1_ACTIVATION"]).lower()

    pool1_size = _as_tuple2(cfg["CNN_POOL1_SIZE"], "CNN_POOL1_SIZE")
    pool1_strides = _as_tuple2(cfg["CNN_POOL1_STRIDES"], "CNN_POOL1_STRIDES")

    conv2_filters = int(cfg["CNN_CONV2_FILTERS"])
    conv2_kernel = _as_tuple2(cfg["CNN_CONV2_KERNEL"], "CNN_CONV2_KERNEL")
    conv2_strides = _as_tuple2(cfg["CNN_CONV2_STRIDES"], "CNN_CONV2_STRIDES")
    conv2_padding = str(cfg["CNN_CONV2_PADDING"]).lower()
    conv2_act = str(cfg["CNN_CONV2_ACTIVATION"]).lower()

    pool2_size = _as_tuple2(cfg["CNN_POOL2_SIZE"], "CNN_POOL2_SIZE")
    pool2_strides = _as_tuple2(cfg["CNN_POOL2_STRIDES"], "CNN_POOL2_STRIDES")

    dense_units = int(cfg["CNN_DENSE_UNITS"])
    dense_act = str(cfg["CNN_DENSE_ACTIVATION"]).lower()
    head_dropout = float(cfg.get("CNN_HEAD_DROPOUT", 0.0))

    out_act = str(cfg.get("CNN_OUTPUT_ACTIVATION", "softmax")).lower()

    inp = layers.Input(shape=(img_size, img_size, in_ch), name="input")

    x = layers.Conv2D(
        conv1_filters, conv1_kernel, strides=conv1_strides, padding=conv1_padding,
        activation=conv1_act, kernel_regularizer=kreg, name="conv1"
    )(inp)
    x = layers.MaxPooling2D(pool_size=pool1_size, strides=pool1_strides, name="pool1")(x)

    x = layers.Conv2D(
        conv2_filters, conv2_kernel, strides=conv2_strides, padding=conv2_padding,
        activation=conv2_act, kernel_regularizer=kreg, name="conv2"
    )(x)
    x = layers.MaxPooling2D(pool_size=pool2_size, strides=pool2_strides, name="pool2")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation=dense_act, kernel_regularizer=kreg, name="dense")(x)
    if head_dropout and head_dropout > 0:
        x = layers.Dropout(head_dropout, name="head_dropout")(x)

    # Saída softmax (dtype float32 p/ estabilidade; importante com mixed precision)
    out = layers.Dense(
        num_classes, activation=out_act, kernel_regularizer=kreg, dtype="float32", name="probs"
    )(x)

    return Model(inputs=inp, outputs=out, name="CNNTraditionalByTheBook")

# DATA LOADING + STRATIFIED KFOLD
# =========================
def load_and_filter_manifest(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Lê manifest, filtra top classes por frequência, remapeia labels -> y em [0..C-1],
    e remove classes com menos de KFOLDS exemplos (necessário para CV estratificado).

    Didática:
    - "manifest.csv" aponta para arquivos (dst) e rótulos (label).
    - Remapear labels é útil porque labels podem ser IDs arbitrários; para loss
      SparseCategoricalCrossentropy precisamos de classes 0..C-1.
    """
    manifest_path = cfg["DATASET_DIR"] / cfg["MANIFEST_NAME"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    need = {"dst", "label"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"manifest must contain columns {need}, got {df.columns.tolist()}")

    # Filtra por ok==True se a coluna existir e ONLY_OK estiver ligado
    if cfg["ONLY_OK"] and "ok" in df.columns:
        df = df[df["ok"] == True].copy()

    df["dst"] = df["dst"].astype(str)
    df["label"] = df["label"].astype(int)

    # Mantém só arquivos que existem no disco (evita quebrar no tf.io.read_file)
    df = df[df["dst"].apply(lambda p: Path(p).exists())].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise RuntimeError("No valid images after filtering by file existence / ok flag.")

    # Top classes por frequência
    top_fraction = cfg["TOP_CLASS_FRACTION"]
    if not (0 < top_fraction <= 1.0):
        raise ValueError("TOP_CLASS_FRACTION must be in (0,1].")

    if top_fraction < 1.0:
        counts = df["label"].value_counts()
        k = max(1, int(math.ceil(top_fraction * len(counts))))
        keep_labels = set(counts.nlargest(k).index.tolist())
        df = df[df["label"].isin(keep_labels)].copy()
        df.reset_index(drop=True, inplace=True)

    # Remapeia labels originais para índices 0..C-1
    uniq = sorted(df["label"].unique().tolist())
    label2idx = {lab: i for i, lab in enumerate(uniq)}
    df["y"] = df["label"].map(label2idx).astype(int)

    # Garante que cada classe tenha pelo menos KFOLDS exemplos
    kfolds = int(cfg["KFOLDS"])
    counts = df["y"].value_counts()
    keep_y = set(counts[counts >= kfolds].index.tolist())
    df = df[df["y"].isin(keep_y)].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise RuntimeError("Dataset empty after class filtering for k-fold.")

    print(f"[INFO] images={len(df)} classes={df['y'].nunique()}")
    return df, label2idx


def stratified_kfold_indices(df: pd.DataFrame, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    K-Fold estratificado:
    - Para cada classe y: embaralha índices e divide em k "chunks".
    - Fold i: validação = chunk i de cada classe; treino = resto.

    Vantagem:
    - Cada fold preserva (aproximadamente) proporções de classes.
    """
    per_y: Dict[int, List[np.ndarray]] = {}
    for y, grp in df.groupby("y", sort=False):
        idx = grp.index.to_numpy()
        rng = np.random.default_rng(seed + int(y))
        rng.shuffle(idx)
        per_y[int(y)] = np.array_split(idx, k)

    all_idx = set(df.index.tolist())
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_idx = np.concatenate([per_y[y][i] for y in per_y], axis=0)
        val_set = set(val_idx.tolist())
        train_idx = np.array(sorted(list(all_idx - val_set)), dtype=np.int64)
        val_idx = np.array(sorted(list(val_set)), dtype=np.int64)
        folds.append((train_idx, val_idx))
    return folds


# =========================
# TF.DATA PIPELINE
# =========================
def make_tf_dataset(df: pd.DataFrame, cfg: Dict[str, Any], training: bool):
    """
    Cria um tf.data.Dataset eficiente a partir de paths e labels.

    Didática da sequência:
    1) from_tensor_slices: cria dataset (path, y)
    2) shuffle (se treino)
    3) map(_load): lê arquivo -> decodifica JPEG -> float32 -> (resize opcional) -> augment (se treino) -> normalize
    4) batch
    5) (cache opcional)
    6) options (determinismo)
    7) prefetch (pipeline assíncrono)
    """
    paths = df["dst"].to_numpy().astype(str)
    labels = df["y"].to_numpy().astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Só faz sentido embaralhar se houver pelo menos 2 exemplos.
    # Se len(df) for 0 ou 1, shuffle é inútil e pode explodir (buffer_size=0).
    if training and len(df) >= 2:
        ds = ds.shuffle(
            buffer_size=min(len(df), 20000),
            seed=int(cfg["SEED"]),
            reshuffle_each_iteration=True,
        )

    img_size = int(cfg["IMG_SIZE"])
    channels = int(cfg["IN_CHANNELS"])
    assume_ingested = bool(cfg.get("ASSUME_INGESTED_SIZE", False))

    # Normalização estilo ImageNet (mean/std por canal)
    mean = tf.constant(cfg["NORM_MEAN"], dtype=tf.float32)[None, None, :]
    std = tf.constant(cfg["NORM_STD"], dtype=tf.float32)[None, None, :]

    dct_method = str(cfg.get("JPEG_DCT_METHOD", "INTEGER_FAST"))

    @tf.function
    def _load(path, y):
        # Lê bytes do arquivo
        raw = tf.io.read_file(path)

        # Decodifica JPEG para tensor [H,W,C]
        img = tf.image.decode_jpeg(raw, channels=channels, dct_method=dct_method)

        # Converte para float32 em [0,1] (importante antes de normalize/augment)
        img = tf.image.convert_image_dtype(img, tf.float32)

        if assume_ingested:
            # Se ingestão garantiu IMG_SIZE x IMG_SIZE, fixamos shape (ajuda o TF a otimizar)
            img = tf.ensure_shape(img, [img_size, img_size, channels])
        else:
            # Caso contrário, fazemos resize (mais lento, mas seguro)
            img = tf.image.resize(img, [img_size, img_size], method="bilinear")
            img = tf.ensure_shape(img, [img_size, img_size, channels])

        # AUGMENTATION (apenas no treino)
        if training:
            if bool(cfg.get("AUG_HFLIP", True)):
                img = tf.image.random_flip_left_right(img)
            pad = int(cfg.get("AUG_PAD", 0))
            if pad > 0:
                # Reflect padding evita bordas "pretas" e preserva continuidade
                img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
                img = tf.image.random_crop(img, size=[img_size, img_size, channels])

        # Normalização: (x - mean) / std (por canal)
        img = (img - mean) / std
        return img, y

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    bs = int(cfg["BATCH_SIZE"])
    ds = ds.batch(bs, drop_remainder=training)

    if bool(cfg.get("CACHE_DATASET", False)):
        ds = ds.cache()

    options = tf.data.Options()
    options.experimental_deterministic = bool(cfg.get("DETERMINISTIC", False))
    ds = ds.with_options(options)

    if bool(cfg.get("PREFETCH", True)):
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =========================
# OPTIMIZER / LOSS
# =========================
def make_learning_rate(cfg: Dict[str, Any], steps_per_epoch: int, epochs_max: int):
    """
    Learning-rate schedule configurável.

    O TF espera decay_steps em "passos" (batches), então convertemos épocas -> passos:
      total_steps ~= steps_per_epoch * epochs

    Tipos implementados:
    - constant
    - exponential_decay
    - cosine_decay
    - cosine_decay_restarts
    """
    base_lr = float(cfg["LR"])
    schedule_type = str(cfg.get("LR_SCHEDULE_TYPE", "constant")).strip().lower()

    if schedule_type in ("constant", "none", ""):
        return base_lr

    if steps_per_epoch <= 0:
        # fallback seguro
        return base_lr

    if schedule_type in ("exponential_decay", "exp", "exponential"):
        decay_epochs = float(cfg.get("LR_DECAY_EPOCHS", 10.0))
        decay_steps = int(max(1, round(steps_per_epoch * decay_epochs)))
        decay_rate = float(cfg.get("LR_DECAY_RATE", 0.96))
        staircase = bool(cfg.get("LR_STAIRCASE", False))
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )

    if schedule_type in ("cosine_decay", "cosine"):
        total_steps = int(max(1, steps_per_epoch * int(max(1, epochs_max))))
        alpha = float(cfg.get("LR_COSINE_ALPHA", 0.0))
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=total_steps,
            alpha=alpha,
        )

    if schedule_type in ("cosine_decay_restarts", "cosine_restart", "cosine_restarts"):
        first_decay_epochs = float(cfg.get("LR_COSINE_FIRST_DECAY_EPOCHS", 10.0))
        first_decay_steps = int(max(1, round(steps_per_epoch * first_decay_epochs)))
        t_mul = float(cfg.get("LR_COSINE_T_MUL", 2.0))
        m_mul = float(cfg.get("LR_COSINE_M_MUL", 1.0))
        alpha = float(cfg.get("LR_COSINE_ALPHA", 0.0))
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=base_lr,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
        )

    raise ValueError(f"Unknown LR_SCHEDULE_TYPE='{cfg.get('LR_SCHEDULE_TYPE')}'")


def make_optimizer(cfg: Dict[str, Any], steps_per_epoch: int, epochs_max: int):
    """
    Otimizador Adam, agora com learning-rate schedule configurável.
    Se MIXED_PRECISION=True, encapsula com LossScaleOptimizer.
    """
    lr = make_learning_rate(cfg, steps_per_epoch=steps_per_epoch, epochs_max=epochs_max)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    if bool(cfg.get("MIXED_PRECISION", False)):
        try:
            opt = mixed_precision.LossScaleOptimizer(opt)
        except Exception as e:
            print("[WARN] LossScaleOptimizer unavailable:", e)
    return opt


def make_loss(cfg: Dict[str, Any], num_classes: int):
    """
    Perda: Categorical Cross-Entropy + saída softmax.

    - Default (USE_SPARSE_CE=False): usa CategoricalCrossentropy (one-hot internamente).
    - Opcional (USE_SPARSE_CE=True): usa SparseCategoricalCrossentropy (equivalente e mais eficiente).
    """
    if bool(cfg.get("USE_SPARSE_CE", False)):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def _loss(y_int, y_pred):
        y_oh = tf.one_hot(tf.cast(y_int, tf.int32), depth=int(num_classes))
        return ce(y_oh, y_pred)

    return _loss

# =========================
# TRAIN / EVAL (FAST + COUNTERS)
# =========================
def train_one_epoch(
    model: Model,
    ds,
    optimizer,
    loss_fn,
    cfg: Dict[str, Any],
    fold_i: int,
    epoch: int,
):
    """
    Treina 1 época em um fold.

    Didática:
    - forward: model(x, training=True) -> probs (softmax)
    - loss: cross-entropy + regularização L2 (model.losses)
    - backward: gradientes via GradientTape
    - apply: optimizer.apply_gradients
    - métricas simples: acurácia e erro (1 - acc)
    """
    log_every = int(cfg.get("LOG_EVERY_N_STEPS", 50))
    xla = bool(cfg.get("XLA", False))

    @tf.function(jit_compile=xla)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            probs = model(x, training=True)

            # Loss base sempre em float32 (estabilidade numérica)
            base_loss = loss_fn(y, tf.cast(probs, tf.float32))
            base_loss = tf.cast(base_loss, tf.float32)

            # Regularização (L2) também em float32
            if model.losses:
                reg_loss = tf.add_n([tf.cast(l, tf.float32) for l in model.losses])
            else:
                reg_loss = tf.constant(0.0, dtype=tf.float32)

            loss = base_loss + reg_loss

        # Mixed precision loss scaling
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) or (
                hasattr(optimizer, "get_scaled_loss") and hasattr(optimizer, "get_unscaled_gradients")
        ):
            scaled_loss = optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, model.trainable_variables)

        # Filtra gradientes None
        grads_and_vars = [(g, v) for (g, v) in zip(grads, model.trainable_variables) if g is not None]
        if not grads_and_vars:
            raise RuntimeError(
                "Todos os gradientes vieram None. Possível overflow/NaN ou perda desconectada do grafo."
            )

        optimizer.apply_gradients(grads_and_vars)

        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(y, tf.int32)), tf.int32))
        batch_n = tf.shape(y)[0]
        return loss, correct, batch_n

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    t_epoch0 = time.perf_counter()
    t_last = t_epoch0

    first_batch_printed = False

    for step, (x, y) in enumerate(ds, start=1):
        loss, correct, batch_n = train_step(x, y)

        bn = int(batch_n.numpy())
        total_loss += float(loss.numpy()) * bn
        total_correct += int(correct.numpy())
        total_seen += bn

        # Debug do primeiro batch (device/dtype)
        if bool(cfg.get("PRINT_FIRST_BATCH_INFO", True)) and not first_batch_printed:
            first_batch_printed = True
            try:
                print(f"[DBG] Fold {fold_i} Epoch {epoch} first batch:")
                print(f"      x.shape={x.shape} x.dtype={x.dtype} x.device={x.device}")
                print(f"      y.shape={y.shape} y.dtype={y.dtype}")
            except Exception:
                pass

        # Log periódico por steps
        if log_every > 0 and (step % log_every == 0):
            now = time.perf_counter()
            dt = now - t_last
            imgs_per_s = (log_every * int(cfg["BATCH_SIZE"])) / max(dt, 1e-9)
            acc = total_correct / max(total_seen, 1)
            print(
                f"[Fold {fold_i}][Epoch {epoch}] step {step} | "
                f"seen={total_seen} | acc={acc:.4f} | "
                f"{imgs_per_s:.1f} img/s | {dt:.2f}s/{log_every} steps"
            )
            t_last = now

    epoch_seconds = time.perf_counter() - t_epoch0
    mean_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    err = 1.0 - acc
    return mean_loss, acc, err, epoch_seconds


def evaluate(model: Model, ds, loss_fn, cfg: Dict[str, Any]):
    """
    Avaliação (val/test) sem treino:
    - model(x, training=False)
    - loss = cross-entropy + regularização
    - métricas: acc e err
    """
    xla = bool(cfg.get("XLA", False))

    @tf.function(jit_compile=xla)
    def val_step(x, y):
        probs = model(x, training=False)

        base_loss = loss_fn(y, tf.cast(probs, tf.float32))
        base_loss = tf.cast(base_loss, tf.float32)

        if model.losses:
            reg_loss = tf.add_n([tf.cast(l, tf.float32) for l in model.losses])
        else:
            reg_loss = tf.constant(0.0, dtype=tf.float32)

        loss = base_loss + reg_loss

        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(y, tf.int32)), tf.int32))
        batch_n = tf.shape(y)[0]
        return loss, correct, batch_n

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in ds:
        loss, correct, batch_n = val_step(x, y)
        bn = int(batch_n.numpy())
        total_loss += float(loss.numpy()) * bn
        total_correct += int(correct.numpy())
        total_seen += bn

    mean_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    err = 1.0 - acc
    return mean_loss, acc, err


# ============================================================
# PÓS-TREINO: "TESTE" + 10 EXEMPLOS + ONE-VS-ALL
# ============================================================
def _preprocess_single_image(path: str, cfg: Dict[str, Any]) -> tf.Tensor:
    """
    Carrega 1 imagem e aplica exatamente o mesmo pré-processamento do pipeline de avaliação
    (sem augment), retornando tensor [H,W,C] float32 normalizado.

    Mantemos a mesma sequência:
    decode_jpeg -> convert_image_dtype -> (ensure_shape ou resize) -> normalize
    """
    img_size = int(cfg["IMG_SIZE"])
    channels = int(cfg["IN_CHANNELS"])
    assume_ingested = bool(cfg.get("ASSUME_INGESTED_SIZE", False))
    dct_method = str(cfg.get("JPEG_DCT_METHOD", "INTEGER_FAST"))

    mean = tf.constant(cfg["NORM_MEAN"], dtype=tf.float32)[None, None, :]
    std = tf.constant(cfg["NORM_STD"], dtype=tf.float32)[None, None, :]

    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=channels, dct_method=dct_method)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if assume_ingested:
        img = tf.ensure_shape(img, [img_size, img_size, channels])
    else:
        img = tf.image.resize(img, [img_size, img_size], method="bilinear")
        img = tf.ensure_shape(img, [img_size, img_size, channels])

    img = (img - mean) / std
    return img


def _one_vs_all_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> pd.DataFrame:
    """
    Calcula matriz de confusão one-vs-all (TP,FP,FN,TN) para cada classe dada.

    Para cada classe c:
    - positivo = (classe == c)
    - negativo = (classe != c)
    """
    rows = []
    for c in classes:
        true_pos = (y_true == c)
        pred_pos = (y_pred == c)

        tp = int(np.sum(true_pos & pred_pos))
        fp = int(np.sum(~true_pos & pred_pos))
        fn = int(np.sum(true_pos & ~pred_pos))
        tn = int(np.sum(~true_pos & ~pred_pos))

        rows.append({"class": c, "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    return pd.DataFrame(rows)



def compute_confusion_matrix(
    model: Model,
    ds,
    num_classes: int,
    cfg: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    Calcula matriz de confusão (y_true x y_pred) no dataset ds.

    Para datasets com muitas classes, isso pode ficar enorme. Por isso:
    - só calcula se SAVE_CONFUSION_MATRIX=True e num_classes <= CONFUSION_MATRIX_MAX_CLASSES.
    """
    if not bool(cfg.get("SAVE_CONFUSION_MATRIX", True)):
        return None

    max_c = int(cfg.get("CONFUSION_MATRIX_MAX_CLASSES", 500))
    if int(num_classes) > max_c:
        print(f"[POST] Confusion matrix pulada: num_classes={num_classes} > CONFUSION_MATRIX_MAX_CLASSES={max_c}")
        return None

    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)

    for batch in ds:
        x, y = batch
        probs = model(x, training=False)
        preds = tf.argmax(probs, axis=-1, output_type=tf.int32)

        y_np = y.numpy().astype(np.int64)
        p_np = preds.numpy().astype(np.int64)

        # Atualiza contagens (vectorizado por índice)
        np.add.at(cm, (y_np, p_np), 1)

    return cm

def post_training_test_report(
    cfg: Dict[str, Any],
    run_dir: Path,
    df_val_as_test: pd.DataFrame,
    num_classes: int,
    label2idx: Dict[int, int],
    best_weights: List[np.ndarray],
    title: str = "Avaliação final",
) -> None:
    """
    Relatório final após o CV:
    - Reconstrói o modelo e aplica os pesos do "melhor fold".
    - Avalia no conjunto "teste" (aqui: validação do melhor fold do K-Fold).
    - Imprime erro/acc/loss e mostra 10 exemplos aleatórios com predições.
    - Gera matriz one-vs-all para esses 10 exemplos.

    Observação didática:
    - Em K-Fold CV, não existe um test set único por padrão.
      O mais honesto, sem mudar sua lógica, é tratar o fold segurado (val) como "teste"
      do melhor fold (held-out).

    Também pode ser usado com um TESTE FINAL real (hold-out antes do CV),
    mantendo a mesma lógica de relatório.
    """
    print("\n" + "=" * 80)
    print(f"[POST] {title}")
    print("=" * 80)

    # Inverso para recuperar "label original" a partir do índice reindexado (y)
    idx2label = {v: k for k, v in label2idx.items()}

    # Dataset de teste, sem augment
    test_ds = make_tf_dataset(df_val_as_test, cfg, training=False)

    # Reconstrói e aplica pesos
    with tf.device(cfg["DEVICE"]):
        model = build_cnn(cfg, num_classes=num_classes)
    model.set_weights(best_weights)

    loss_fn = make_loss(cfg, num_classes)
    te_loss, te_acc, te_err = evaluate(model, test_ds, loss_fn, cfg)

    print(f"[TEST] loss={te_loss:.6f} acc={te_acc:.6f} err={te_err:.6f}")
    print(f"[TEST] n={len(df_val_as_test)} classes={num_classes} device={cfg['DEVICE']}")

    # --------------------------------------------------------
    # Matriz de confusão no conjunto de teste
    # --------------------------------------------------------
    cm = compute_confusion_matrix(model, test_ds, num_classes=num_classes, cfg=cfg)
    if cm is not None:
        try:
            cm_path = run_dir / "confusion_matrix_test.csv"
            pd.DataFrame(cm).to_csv(cm_path, index=False, encoding="utf-8")
            print(f"[SAVED] {cm_path}")
        except Exception as e:
            print("[WARN] Could not save confusion matrix:", e)


    # --------------------------------------------------------
    # 10 exemplos aleatórios de predição
    # --------------------------------------------------------
    n_show = min(10, len(df_val_as_test))
    if n_show <= 0:
        print("[POST] Sem exemplos para mostrar (df_test vazio).")
        return

    sample_df = df_val_as_test.sample(n=n_show, random_state=int(cfg["SEED"])).reset_index(drop=True)

    y_true_list = []
    y_pred_list = []

    print("\n[POST] 10 exemplos aleatórios (identificação):")
    for i in range(n_show):
        p = str(sample_df.loc[i, "dst"])
        y_true = int(sample_df.loc[i, "y"])
        label_true_original = int(idx2label.get(y_true, -1))

        img = _preprocess_single_image(p, cfg)
        x = tf.expand_dims(img, axis=0)  # [1,H,W,C]

        # Predição (saída já é softmax)
        probs = model(x, training=False)
        probs = tf.cast(probs, tf.float32)[0].numpy()

        y_pred = int(np.argmax(probs))
        label_pred_original = int(idx2label.get(y_pred, -1))
        conf = float(probs[y_pred])

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

        print(
            f"  [{i+1:02d}] path={p}\n"
            f"       y_true={y_true} (label_original={label_true_original}) | "
            f"y_pred={y_pred} (label_original={label_pred_original}) | "
            f"conf={conf:.4f}"
        )

    y_true_arr = np.array(y_true_list, dtype=np.int64)
    y_pred_arr = np.array(y_pred_list, dtype=np.int64)

    # Classes presentes nesses 10 exemplos (ou preditas neles)
    classes_present = sorted(list(set(y_true_arr.tolist()) | set(y_pred_arr.tolist())))

    # --------------------------------------------------------
    # Matriz de confusão one-vs-all para esses 10 exemplos
    # --------------------------------------------------------
    ova_df = _one_vs_all_confusion(y_true_arr, y_pred_arr, classes_present)

    print("\n[POST] Matriz de confusão one-vs-all (apenas para os 10 exemplos acima):")
    print(ova_df.to_string(index=False))

    # Opcional: salvar relatórios no run_dir
    try:
        ova_path = run_dir / "post_one_vs_all_10examples.csv"
        ova_df.to_csv(ova_path, index=False, encoding="utf-8")
        preds_path = run_dir / "post_predictions_10examples.csv"
        out_pred = sample_df.copy()
        out_pred["y_pred"] = y_pred_arr
        out_pred.to_csv(preds_path, index=False, encoding="utf-8")
        print(f"\n[SAVED] {ova_path}")
        print(f"[SAVED] {preds_path}")
    except Exception as e:
        print("[WARN] Could not save post-training reports:", e)


# =========================
# Salvar pesos em TXT
# =========================
def save_model_weights_txt(model: Model, path: Path) -> None:
    """
    Salva os pesos do modelo em formato texto.

    Observação:
    - Isso pode gerar um arquivo grande, mas é útil para auditoria/debug.
    - Aqui imprimimos CADA tensor completo (sem truncar), com name/shape/dtype.
    """
    weights = model.get_weights()
    vars_ = model.weights  # inclui trainable e non-trainable (BN moving stats etc.)

    ensure_dir(path.parent)

    # Em geral, len(vars_) == len(weights), mas por segurança lidamos com discrepâncias.
    n = min(len(weights), len(vars_))

    with path.open("w", encoding="utf-8") as f:
        f.write("# Model weights dump (full)\n")
        f.write(f"# model_name={model.name}\n")
        f.write(f"# num_tensors={len(weights)}\n")
        f.write(f"# written_tensors={n}\n\n")

        with np.printoptions(threshold=np.inf, linewidth=200, suppress=False):
            for i in range(n):
                var = vars_[i]
                arr = weights[i]
                f.write(f"\n# [{i}] {var.name} | shape={arr.shape} | dtype={arr.dtype}\n")
                f.write(np.array2string(arr, separator=", "))
                f.write("\n")

            if len(weights) > n:
                f.write("\n# [WARN] Há mais pesos em get_weights() do que em model.weights; não pareados.\n")
                for j in range(n, len(weights)):
                    arr = weights[j]
                    f.write(f"\n# [extra {j}] shape={arr.shape} | dtype={arr.dtype}\n")
                    f.write(np.array2string(arr, separator=", "))
                    f.write("\n")


# =========================
# K-FOLD CV RUNNER
# =========================
def run_kfold_cv(cfg: Dict[str, Any]) -> Tuple[float, Path]:
    """
    Executa K-Fold CV e grava arquivos de log.
    Retorna: (cv_mean_best_val_err, run_dir)

    Além disso:
    - Guarda o "melhor fold" (menor val_err) e faz um relatório final nele.
    - Antes do CV, cria um TESTE FINAL (hold-out) com % de classes configurável.
    - Após o CV, usa o melhor modelo do CV como inicialização e faz um treino final
      (com val interno + early stopping) e avalia no TESTE FINAL.
    """
    setup_tf_performance(cfg)
    set_seed(int(cfg["SEED"]))

    df_full, label2idx = load_and_filter_manifest(cfg)

    # Confirma resolução como ingerida (antes de resize)
    confirm_image_resolution(df_full, cfg)

    # --------------------------------------------------------
    # Split TESTE FINAL antes do CV
    # --------------------------------------------------------
    cv_df, final_test_df, test_classes = stratified_holdout_split_by_class(
        df=df_full,
        cfg=cfg,
        holdout_class_fraction=float(cfg.get("FINAL_TEST_CLASS_FRACTION", 1.0)),
        holdout_fraction_per_class=float(cfg.get("FINAL_TEST_FRACTION_PER_CLASS", 0.0)),
        min_keep_per_class=int(cfg["KFOLDS"]),     # sobrar KFOLDS p/ CV estratificado
        seed_offset=900_000,
    )

    print("\n" + "=" * 80)
    print("[INFO] FINAL TEST split (before CV)")
    print("=" * 80)
    print(f"[INFO] FINAL_TEST_CLASS_FRACTION={cfg.get('FINAL_TEST_CLASS_FRACTION')} "
          f"-> classes escolhidas={len(test_classes)}/{df_full['y'].nunique()}")
    print(f"[INFO] FINAL_TEST_FRACTION_PER_CLASS={cfg.get('FINAL_TEST_FRACTION_PER_CLASS')} "
          f"-> final_test_images={len(final_test_df)} | cv_images={len(cv_df)}")

    if len(final_test_df) == 0:
        print("[WARN] final_test_df ficou vazio. Ajuste FINAL_TEST_* para reservar imagens de teste.")
    else:
        # Diagnóstico rápido do teste final:
        print(f"[INFO] final_test classes presentes={final_test_df['y'].nunique()}")

    # --------------------------------------------------------
    # CV propriamente dito (em cima de cv_df)
    # --------------------------------------------------------
    k = int(cfg["KFOLDS"])
    folds = stratified_kfold_indices(cv_df, k=k, seed=int(cfg["SEED"]))
    num_classes = int(cv_df["y"].nunique())

    stamp = now_stamp()
    run_dir = cfg["DATASET_DIR"] / cfg["RUNS_DIRNAME"] / stamp
    ensure_dir(run_dir)

    save_json(cfg, run_dir / "hyperparameters.json")
    save_json({str(k): int(v) for k, v in label2idx.items()}, run_dir / "label2idx.json")


    # --------------------------------------------------------
    # (NOVO) Persistir o HOLDOUT FINAL (final_test_df) e o pool de CV (cv_df)
    #        no run_dir, para que o script de avaliação reutilize exatamente
    #        o mesmo holdout e elimine risco de data leakage.
    #
    # Arquivos gerados:
    # - final_test_manifest.csv  -> apenas exemplos do holdout FINAL TEST
    # - cv_manifest.csv          -> pool restante (usado no CV/treino)
    # --------------------------------------------------------
    try:
        final_test_manifest = run_dir / "final_test_manifest.csv"
        cv_manifest = run_dir / "cv_manifest.csv"

        # Salvar apenas colunas essenciais (mantém compatibilidade com carregar_dataset_manifest_csv)
        cols = [c for c in ("dst", "label", "y", "ok") if c in final_test_df.columns]
        final_test_df[cols].to_csv(final_test_manifest, index=False, encoding="utf-8")

        cols_cv = [c for c in ("dst", "label", "y", "ok") if c in cv_df.columns]
        cv_df[cols_cv].to_csv(cv_manifest, index=False, encoding="utf-8")

        print(f"[INFO] Holdout FINAL TEST salvo em: {final_test_manifest}")
        print(f"[INFO] Pool de CV salvo em: {cv_manifest}")
    except Exception as e:
        print(f"[WARN] Falha ao salvar manifests de split (holdout/CV): {e}")

    errors_csv = run_dir / "run_errors.csv"
    with errors_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "fold", "epoch",
            "train_loss", "train_acc", "train_err",
            "val_loss", "val_acc", "val_err",
            "epoch_seconds",
        ])

        fold_summaries = []
        best_val_errs = []

        # Guardar o melhor fold global para "teste" final
        best_global_val_err = float("inf")
        best_global_weights: Optional[List[np.ndarray]] = None
        best_global_val_df: Optional[pd.DataFrame] = None
        best_global_fold_i: int = -1
        best_global_epoch: int = -1

        # Early stopping (CV)
        patience_cv = int(cfg.get("EARLY_STOPPING_PATIENCE_CV", 5))
        min_delta = float(cfg.get("EARLY_STOPPING_MIN_DELTA", 0.0))

        epochs_cv = int(cfg.get("EPOCHS_CV", cfg.get("EPOCHS", 50)))

        for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
            # Importante em CV: limpar grafo entre folds
            tf.keras.backend.clear_session()
            set_seed(int(cfg["SEED"]) + 1000 * fold_i)

            train_df = cv_df.iloc[train_idx].reset_index(drop=True)
            val_df = cv_df.iloc[val_idx].reset_index(drop=True)

            train_ds = make_tf_dataset(train_df, cfg, training=True)
            val_ds = make_tf_dataset(val_df, cfg, training=False)

            # Cria modelo no device desejado
            with tf.device(cfg["DEVICE"]):
                model = build_cnn(cfg, num_classes=num_classes)

            # Optimizer recebe steps_per_epoch e epochs_max (para o schedule)
            steps_per_epoch = int(math.ceil(len(train_df) / max(1, int(cfg["BATCH_SIZE"]))))
            optimizer = make_optimizer(cfg, steps_per_epoch=steps_per_epoch, epochs_max=epochs_cv)

            loss_fn = make_loss(cfg, num_classes)

            best_val_err = float("inf")
            best_epoch = -1
            best_weights = None
            no_improve = 0

            print(f"\n[INFO] Fold {fold_i}/{k} | train={len(train_df)} val={len(val_df)} | classes={num_classes}")
            print(f"[INFO] Device scope: {cfg['DEVICE']}")
            print(f"[INFO] LR schedule: {cfg.get('LR_SCHEDULE_TYPE', 'constant')} | steps/epoch={steps_per_epoch}")

            for epoch in range(1, epochs_cv + 1):
                tr_loss, tr_acc, tr_err, dt = train_one_epoch(
                    model, train_ds, optimizer, loss_fn, cfg, fold_i, epoch
                )
                va_loss, va_acc, va_err = evaluate(model, val_ds, loss_fn, cfg)

                w.writerow([
                    fold_i, epoch,
                    f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_err:.6f}",
                    f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_err:.6f}",
                    f"{dt:.3f}",
                ])
                f.flush()

                print(
                    f"[Fold {fold_i}][Epoch {epoch}/{epochs_cv}] "
                    f"train_loss={tr_loss:.4f} train_err={tr_err:.4f} | "
                    f"val_loss={va_loss:.4f} val_err={va_err:.4f} | {dt:.1f}s"
                )

                # Mantém melhor época do fold + early stopping
                improved = (va_err < (best_val_err - min_delta))
                if improved:
                    best_val_err = float(va_err)
                    best_epoch = int(epoch)
                    best_weights = model.get_weights()
                    no_improve = 0
                else:
                    no_improve += 1
                    if patience_cv > 0 and no_improve >= patience_cv:
                        print(
                            f"[EARLY STOP][Fold {fold_i}] sem melhora em val_err por {patience_cv} épocas "
                            f"(best_epoch={best_epoch}, best_val_err={best_val_err:.6f})."
                        )
                        break

            # Restaura pesos do melhor epoch do fold
            if best_weights is not None:
                model.set_weights(best_weights)

            best_val_errs.append(best_val_err)
            fold_summaries.append({
                "fold": fold_i,
                "best_epoch": best_epoch,
                "best_val_err": best_val_err,
            })

            # Guarda o melhor fold global para o relatório final
            if best_val_err < best_global_val_err and best_weights is not None:
                best_global_val_err = float(best_val_err)
                best_global_weights = best_weights
                best_global_val_df = val_df.copy()
                best_global_fold_i = int(fold_i)
                best_global_epoch = int(best_epoch)

            # Salva modelo (opcional)
            if bool(cfg.get("SAVE_MODEL", False)):
                models_dir = run_dir / "models"
                ensure_dir(models_dir)
                save_path = models_dir / f"model_fold{fold_i}_best.keras"
                model.save(save_path)
                print(f"[SAVED] {save_path}")

    mean_err = float(np.mean(best_val_errs)) if best_val_errs else float("nan")
    std_err = float(np.std(best_val_errs, ddof=1)) if len(best_val_errs) > 1 else 0.0

    summary = {
        "kfolds": k,
        "num_images_full": int(len(df_full)),
        "num_images_cv": int(len(cv_df)),
        "num_images_final_test": int(len(final_test_df)),
        "num_classes": int(num_classes),
        "final_test_classes_chosen": int(len(test_classes)),
        "best_val_errs": best_val_errs,
        "cv_mean_best_val_err": mean_err,
        "cv_std_best_val_err": std_err,
        "folds": fold_summaries,
        "best_global_fold": int(best_global_fold_i),
        "best_global_epoch": int(best_global_epoch),
        "best_global_val_err": float(best_global_val_err) if math.isfinite(best_global_val_err) else None,
        "run_dir": str(run_dir),
        "device": cfg["DEVICE"],
        "timestamp": stamp,
        "mixed_precision": bool(cfg.get("MIXED_PRECISION", False)),
        "xla": bool(cfg.get("XLA", False)),
        "assume_ingested_size": bool(cfg.get("ASSUME_INGESTED_SIZE", False)),
        "lr_schedule_type": str(cfg.get("LR_SCHEDULE_TYPE", "constant")),
    }
    save_json(summary, run_dir / "summary.json")

    print("\n[DONE] CV finished.")
    print(f"[RESULT] mean(best_val_err)={mean_err:.6f} | std={std_err:.6f}")
    print(f"[FILES] {run_dir}")

    # --------------------------------------------------------
    # Ajuste antigo: relatório final no melhor fold (val como "teste")
    # --------------------------------------------------------
    try:
        if best_global_weights is not None and best_global_val_df is not None:
            print(f"\n[POST] Melhor fold global: fold={best_global_fold_i} best_val_err={best_global_val_err:.6f}")
            post_training_test_report(
                cfg=cfg,
                run_dir=run_dir,
                df_val_as_test=best_global_val_df,
                num_classes=num_classes,
                label2idx=label2idx,
                best_weights=best_global_weights,
                title="Avaliação no VAL do melhor fold (mantido, para comparabilidade)",
            )
        else:
            print("[WARN] Could not run post-training report: best fold state not available.")
    except Exception as e:
        print("[WARN] Post-training report failed:", e)

    # --------------------------------------------------------
    # TREINO FINAL (inicializa com melhor modelo do CV) + TESTE FINAL
    # --------------------------------------------------------
    try:
        if best_global_weights is None:
            print("[WARN] Sem best_global_weights; pulando treino final/teste final.")
            return mean_err, run_dir

        if len(final_test_df) == 0:
            print("[WARN] final_test_df vazio; pulando teste final.")
            return mean_err, run_dir

        # Split interno (por classe) para validação do treino final
        final_train_df, final_val_df, _ = stratified_holdout_split_by_class(
            df=cv_df,
            cfg=cfg,
            holdout_class_fraction=1.0,  # aqui queremos val para TODAS as classes do treino final
            holdout_fraction_per_class=float(cfg.get("FINAL_TRAIN_VAL_FRACTION_PER_CLASS", 0.0)),
            min_keep_per_class=1,        # ao menos 1 exemplo por classe no treino final
            seed_offset=910_000,
        )

        print("\n" + "=" * 80)
        print("[INFO] FINAL TRAIN split (before FINAL TEST)")
        print("=" * 80)
        print(f"[INFO] FINAL_TRAIN_VAL_FRACTION_PER_CLASS={cfg.get('FINAL_TRAIN_VAL_FRACTION_PER_CLASS')}")
        print(f"[INFO] final_train_images={len(final_train_df)} | final_val_images={len(final_val_df)}")
        print(f"[INFO] final_test_images={len(final_test_df)}")

        final_train_ds = make_tf_dataset(final_train_df, cfg, training=True)
        final_val_ds = make_tf_dataset(final_val_df, cfg, training=False)
        final_test_ds = make_tf_dataset(final_test_df, cfg, training=False)

        # Reconstrói o modelo e inicializa com pesos do melhor fold do CV
        tf.keras.backend.clear_session()
        set_seed(int(cfg["SEED"]) + 777_777)

        with tf.device(cfg["DEVICE"]):
            final_model = build_cnn(cfg, num_classes=num_classes)
        final_model.set_weights(best_global_weights)

        loss_fn = make_loss(cfg, num_classes)

        # Diagnóstico: avalia no teste final ANTES do treino final (apenas para referência)
        pre_loss, pre_acc, pre_err = evaluate(final_model, final_test_ds, loss_fn, cfg)
        print("\n" + "-" * 80)
        print("[FINAL TEST] (antes do treino final) usando pesos do melhor fold do CV")
        print("-" * 80)
        print(f"[FINAL TEST][PRE] loss={pre_loss:.6f} acc={pre_acc:.6f} err={pre_err:.6f}")

        epochs_final = int(cfg.get("EPOCHS_FINAL", 0))
        patience_final = int(cfg.get("EARLY_STOPPING_PATIENCE_FINAL", 5))
        min_delta = float(cfg.get("EARLY_STOPPING_MIN_DELTA", 0.0))

        # Optimizer com schedule baseado no treino final
        steps_per_epoch_final = int(math.ceil(len(final_train_df) / max(1, int(cfg["BATCH_SIZE"]))))
        optimizer = make_optimizer(cfg, steps_per_epoch=steps_per_epoch_final, epochs_max=max(1, epochs_final))

        best_final_val_err = float("inf")
        best_final_epoch = -1
        best_final_weights = None
        no_improve = 0

        if epochs_final > 0 and len(final_val_df) > 0 and len(final_train_df) > 0:
            print("\n" + "-" * 80)
            print("[INFO] Treino final (inicializado do melhor CV) + early stopping")
            print("-" * 80)
            print(f"[INFO] EPOCHS_FINAL={epochs_final} patience_final={patience_final}")
            print(f"[INFO] LR schedule: {cfg.get('LR_SCHEDULE_TYPE', 'constant')} | steps/epoch={steps_per_epoch_final}")

            for epoch in range(1, epochs_final + 1):
                # Reutiliza train_one_epoch para manter a mesma lógica de treino/telemetria.
                tr_loss, tr_acc, tr_err, dt = train_one_epoch(
                    final_model, final_train_ds, optimizer, loss_fn, cfg, fold_i=0, epoch=epoch
                )
                va_loss, va_acc, va_err = evaluate(final_model, final_val_ds, loss_fn, cfg)

                print(
                    f"[FINAL][Epoch {epoch}/{epochs_final}] "
                    f"train_loss={tr_loss:.4f} train_err={tr_err:.4f} | "
                    f"val_loss={va_loss:.4f} val_err={va_err:.4f} | {dt:.1f}s"
                )

                improved = (va_err < (best_final_val_err - min_delta))
                if improved:
                    best_final_val_err = float(va_err)
                    best_final_epoch = int(epoch)
                    best_final_weights = final_model.get_weights()
                    no_improve = 0
                else:
                    no_improve += 1
                    if patience_final > 0 and no_improve >= patience_final:
                        print(
                            f"[EARLY STOP][FINAL] sem melhora em val_err por {patience_final} épocas "
                            f"(best_epoch={best_final_epoch}, best_val_err={best_final_val_err:.6f})."
                        )
                        break

            if best_final_weights is not None:
                final_model.set_weights(best_final_weights)
        else:
            print("[WARN] Treino final pulado (EPOCHS_FINAL==0 ou split de final_val/final_train vazio).")

        # Avaliação final no teste final
        te_loss, te_acc, te_err = evaluate(final_model, final_test_ds, loss_fn, cfg)
        print("\n" + "-" * 80)
        print("[FINAL TEST] (após treino final) melhor modelo final")
        print("-" * 80)
        print(f"[FINAL TEST][POST] loss={te_loss:.6f} acc={te_acc:.6f} err={te_err:.6f}")

        # Salva métricas do treino final/teste final
        final_results = {
            "final_train_images": int(len(final_train_df)),
            "final_val_images": int(len(final_val_df)),
            "final_test_images": int(len(final_test_df)),
            "pre_test_loss": float(pre_loss),
            "pre_test_acc": float(pre_acc),
            "pre_test_err": float(pre_err),
            "final_best_val_err": float(best_final_val_err) if math.isfinite(best_final_val_err) else None,
            "final_best_epoch": int(best_final_epoch),
            "post_test_loss": float(te_loss),
            "post_test_acc": float(te_acc),
            "post_test_err": float(te_err),
        }
        save_json(final_results, run_dir / "final_test_results.json")

        # Relatório pós-treino usando a função existente, agora no TESTE FINAL real
        try:
            post_training_test_report(
                cfg=cfg,
                run_dir=run_dir,
                df_val_as_test=final_test_df,
                num_classes=num_classes,
                label2idx=label2idx,
                best_weights=final_model.get_weights(),
                title="Avaliação no TESTE FINAL (hold-out antes do CV)",
            )
        except Exception as e:
            print("[WARN] post_training_test_report on FINAL TEST failed:", e)

        # Salvar modelo final + pesos txt
        if bool(cfg.get("SAVE_FINAL_MODEL", True)):
            models_dir = run_dir / "models"
            ensure_dir(models_dir)

            final_model_path = models_dir / str(cfg.get("FINAL_MODEL_FILENAME", "final_model_best.keras"))
            final_weights_txt = models_dir / str(cfg.get("FINAL_WEIGHTS_TXT_FILENAME", "final_model_best_weights.txt"))

            final_model.save(final_model_path)
            print(f"[SAVED] {final_model_path}")

            save_model_weights_txt(final_model, final_weights_txt)
            print(f"[SAVED] {final_weights_txt}")

    except Exception as e:
        print("[WARN] Final train/test stage failed:", e)

    return mean_err, run_dir


# =========================
# MAIN
# =========================
def main() -> float:
    """
    Treina CNN tradicional (by the book) com K-Fold CV.
    Retorna o erro médio (val_err) dos melhores epochs por fold.
    """
    print("[INFO] Using device:", CONFIG["DEVICE"])
    mean_err, _run_dir = run_kfold_cv(CONFIG)
    return mean_err


# ===========================================================================
# REFERÊNCIAS (para citação acadêmica)
# ===========================================================================
# [LeCun98] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
#          Gradient-based learning applied to document recognition.
#          Proceedings of the IEEE.
# [He15]   He, K., Zhang, X., Ren, S., & Sun, J. (2015).
#          Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
#          arXiv:1502.01852.
# [Ioffe15] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training...
#          arXiv:1502.03167.
# [Srivastava14] Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
#          JMLR 15.
# [Krogh91] Krogh, A., & Hertz, J. A. (1991). A Simple Weight Decay Can Improve Generalization.
#          NeurIPS.
# [KroghHertz92] Versão estendida/relacionada frequentemente citada como 1992 em bibliografias.
# [KingmaBa14] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
#          arXiv:1412.6980.
# [LoshchilovHutter16] Loshchilov, I., & Hutter, F. (2016/2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
#          arXiv:1608.03983 (ICLR 2017).
# [Prechelt97] Prechelt, L. (1997). Early Stopping — But When?
#          (técnicas de parada via validação; frequentemente citado em compilações/ebooks).
# [Stone74] Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions.
#          JRSS Series B.
# [Kohavi95] Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.
#          IJCAI.
# [Russakovsky15] Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge.
#          IJCV.
# [Shorten19] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning.
#          Journal of Big Data.
# [Micikevicius17] Micikevicius, P. et al. (2017). Mixed Precision Training.
#          arXiv:1710.03740.
# ===========================================================================

if __name__ == "__main__":
    main()
