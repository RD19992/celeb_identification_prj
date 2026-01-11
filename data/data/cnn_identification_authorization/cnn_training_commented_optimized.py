# -*- coding: utf-8 -*-
"""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Treinamento de CNN (ResNet "explicit") com K-Fold CV (sem sklearn).

VERSÃO OTIMIZADA (GPU):
- tf.data mais rápido (decode JPEG rápido, resize opcional, non-deterministic opcional)
- mixed precision (opcional)
- XLA/jit_compile nos passos de treino/val (opcional)
- contadores/telemetria por step (para debugar CV)
- confirmação da resolução REAL ingerida (sem resize), por amostragem

Arquivos gerados (por execução), dentro de:
  <DATASET_DIR>/runs/<YYYYMMDD_HHMMSS>/

- run_errors.csv            (métricas por fold/época)
- summary.json              (agregados e melhores erros por fold)
- hyperparameters.json      (CONFIG usado na execução)
- label2idx.json            (mapeamento label_original -> y_reindexado)
- [opcional] models/*.keras (salvo apenas se SAVE_MODEL=True)

Obs.: SAVE_MODEL começa DESABILITADO por padrão.

------------------------------------------------------------
AJUSTES PEDIDOS (SEM ALTERAR HIPERPARÂMETROS/LÓGICA):
1) Comentários didáticos PT-BR em cada etapa.
2) Opção configurável para forçar GPU (e opcionalmente falhar se não houver).
3) Pós-treino: avaliação em "teste" (aqui: validação do melhor fold do CV),
   erro, 10 exemplos aleatórios e matriz de confusão one-vs-all desses 10.
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
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import os

# ============================================================
# (1) SELEÇÃO DE ADAPTADOR / BACKEND (DirectML-friendly)
# ============================================================
# Este env var é típico do plugin DirectML: você pode "esconder"
# outros adaptadores e deixar só o índice 0 visível.
# (Mantido como estava, só comentado.)
os.environ["DML_VISIBLE_DEVICES"] = "0"  # mantém apenas o adapter 0 (ex.: GPU discreta)

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras import mixed_precision

# Por padrão, você está fixando a política global em float32.
# Isso evita surpresas de dtype e costuma ser mais estável em ambientes variados.
# (Mantido como estava.)
mixed_precision.set_global_policy("float32")


# ============================================================
# CONFIGURAÇÕES
# ============================================================
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
    "TOP_CLASS_FRACTION": 0.01,       # top fração de classes mais frequentes (mantido)
    "KFOLDS": 2,
    "SEED": 42,

    # --------------------------------------------------------
    # Entrada
    # --------------------------------------------------------
    "IMG_SIZE": 128,
    "IN_CHANNELS": 3,
    "NORM_MEAN": (0.485, 0.456, 0.406),
    "NORM_STD":  (0.229, 0.224, 0.225),

    # --------------------------------------------------------
    # Aumentação de dados (explícita, mínima)
    # --------------------------------------------------------
    "AUG_HFLIP": True,
    "AUG_PAD": 4,                    # reflect-pad + random crop; 0 desabilita

    # --------------------------------------------------------
    # ResNet mínima (explícita)
    # --------------------------------------------------------
    "RES_LAYERS": [1, 1, 1],          # blocks por estágio
    "RES_CHANNELS": [32, 64, 128],    # largura (canais) por estágio
    "USE_BN": True,                  # BatchNorm (normalização em batch)
    "ACTIVATION": "relu",
    "BLOCK_DROPOUT": 0.0,

    # --------------------------------------------------------
    # Regularização
    # --------------------------------------------------------
    "L2_WEIGHT": 1e-4,                # L2 no kernel de Conv/Dense

    # --------------------------------------------------------
    # Treino
    # --------------------------------------------------------
    "BATCH_SIZE": 16,
    "EPOCHS": 10,
    "LR": 1e-3,

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
    "MIXED_PRECISION": False,          # tente True; se backend for "chato", deixe False
    "XLA": False,                      # jit_compile=True nos passos
    "ALLOW_TF32": False,               # CUDA Ampere+: acelera matmul com TF32

    # Debug / contadores
    "LOG_EVERY_N_STEPS": 50,
    "PRINT_FIRST_BATCH_INFO": True,

    # Confirmação de resolução (lê header JPEG; sem resize)
    "RESOLUTION_SAMPLE_N": 256,
    "STRICT_RESOLUTION_CHECK": False,  # True => falha se achar imagem fora do alvo

    # Saídas
    "RUNS_DIRNAME": "runs",
    "SAVE_MODEL": False,              # começa desabilitado

    # --------------------------------------------------------
    # (2) NOVO: FORÇAR GPU / CPU (opcional) - sem mexer no resto
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

# Observação importante:
# - CONFIG["DEVICE"] será definido *depois* que configurarmos os dispositivos,
#   para respeitar FORCE_GPU/FORCE_CPU. Isso não muda os hiperparâmetros nem
#   a lógica do treino; só controla onde o TF tenta alocar operações.


# ============================================================
# GPU/CPU SETUP (com opção de "forçar" GPU)
# ============================================================
def _print_visible_devices() -> None:
    """Imprime dispositivos visíveis (útil para diagnosticar 'estou caindo na CPU')."""
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
    - Aqui, a gente tenta *guiar* o TF: visibilidade de devices e device scope.

    Importante:
    - set_visible_devices deve ser feito cedo (antes do uso pesado de GPU).
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

    # Comportamento padrão (igual ao seu original): usa GPU se existir, senão CPU.
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


# Configura e grava o device escolhido no CONFIG (sem mexer em hiperparâmetros)
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
# RESNET (EXPLICIT)
# =========================
def conv_bn_act(
    x,
    filters: int,
    k: int,
    stride: int,
    use_bn: bool,
    activation: str | None,
    l2_weight: float,
    name: str,
):
    """
    Bloco didático: Conv2D -> (BN) -> (Ativação)

    Convolução 2D (Conv2D) - explicação rápida e útil:
    - Um filtro (kernel) kxk "desliza" sobre a imagem (feature map) e calcula
      combinações lineares locais. Isso permite detectar padrões locais como
      bordas, texturas e, em camadas profundas, partes de objetos/rostos.
    - filters = número de kernels (ou "canais de saída") aprendidos.
      Mais filters => mais capacidade de representar padrões.
    - stride = passo do deslizamento:
      stride=1 preserva mais resolução espacial.
      stride=2 reduz a resolução (downsample) e aumenta campo receptivo efetivo.
    - padding="same" mantém dimensões espaciais (aprox.) quando stride=1
      e controla como bordas são tratadas.
    - use_bias:
      se usamos BatchNorm, normalmente removemos bias do Conv, porque BN já
      aprende um deslocamento (beta) e escala (gamma).
    """
    x = layers.Conv2D(
        filters, k, strides=stride, padding="same", use_bias=not use_bn,
        kernel_regularizer=regularizers.l2(l2_weight),
        name=f"{name}_conv{k}x{k}",
    )(x)

    # Batch Normalization:
    # - estabiliza distribuições internas (por batch), melhora gradientes,
    #   acelera convergência e costuma permitir taxas de aprendizado mais estáveis.
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)

    # Ativação (ReLU por padrão):
    # - introduz não-linearidade, permitindo a rede modelar funções complexas.
    if activation is not None:
        x = layers.Activation(activation, name=f"{name}_{activation}")(x)

    return x


def basic_res_block(
    x,
    filters: int,
    stride: int,
    use_bn: bool,
    activation: str,
    dropout_p: float,
    l2_weight: float,
    name: str,
):
    """
    BasicBlock (ResNet "clássica" simplificada):
      (3x3 -> BN -> Act) -> (3x3 -> BN) + skip -> Act

    Ideia central da ResNet (residual learning):
    - Em vez de aprender uma função direta F(x), o bloco aprende um "resíduo" R(x)
      e soma de volta no input: y = x + R(x).
    - Isso melhora o fluxo de gradientes e facilita treinar redes mais profundas
      (reduz o problema de degradação ao aumentar profundidade).

    Skip connection (atalho):
    - Se stride!=1 (downsample) ou canais mudam, precisamos ajustar o atalho
      com uma Conv 1x1 (projeção) para igualar forma (H,W,C).
    """
    shortcut = x
    in_ch = x.shape[-1]
    out_ch = filters

    # Primeiro conv:
    # - pode reduzir resolução (stride=2) em início de estágio
    y = conv_bn_act(
        x, filters=filters, k=3, stride=stride,
        use_bn=use_bn, activation=activation, l2_weight=l2_weight,
        name=f"{name}_c1",
    )

    # Segundo conv:
    # - mantém resolução, termina o resíduo antes da soma
    y = conv_bn_act(
        y, filters=filters, k=3, stride=1,
        use_bn=use_bn, activation=None, l2_weight=l2_weight,
        name=f"{name}_c2",
    )

    # Dropout espacial (opcional):
    # - zera canais inteiros (mais apropriado em convnets do que dropout "pixel a pixel")
    if dropout_p and dropout_p > 0:
        y = layers.SpatialDropout2D(dropout_p, name=f"{name}_drop")(y)

    # Ajuste do atalho caso dimensões/canais não batam
    if stride != 1 or (in_ch is not None and int(in_ch) != int(out_ch)):
        shortcut = layers.Conv2D(
            out_ch, 1, strides=stride, padding="same", use_bias=not use_bn,
            kernel_regularizer=regularizers.l2(l2_weight),
            name=f"{name}_skip_conv1x1",
        )(shortcut)
        if use_bn:
            shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    # Soma residual + ativação final
    out = layers.Add(name=f"{name}_add")([shortcut, y])
    out = layers.Activation(activation, name=f"{name}_out_{activation}")(out)
    return out


def make_stage(
    x,
    filters: int,
    n_blocks: int,
    first_stride: int,
    use_bn: bool,
    activation: str,
    dropout_p: float,
    l2_weight: float,
    name: str,
):
    """
    Um "estágio" da ResNet = sequência de BasicBlocks com mesma largura (filters),
    onde o primeiro bloco do estágio pode fazer downsample (first_stride=2).

    Didática: como "escalar" a ResNet sem mudar a lógica:
    - Mais profundidade: aumente RES_LAYERS (mais blocos por estágio).
    - Mais largura: aumente RES_CHANNELS (mais filters por estágio).
    - Mais estágios: estender listas RES_LAYERS/RES_CHANNELS (ex.: adicionar stage4).
    Observação: aqui manteremos exatamente os valores do seu CONFIG.
    """
    x = basic_res_block(
        x, filters=filters, stride=first_stride,
        use_bn=use_bn, activation=activation,
        dropout_p=dropout_p, l2_weight=l2_weight,
        name=f"{name}_b1",
    )
    for i in range(2, n_blocks + 1):
        x = basic_res_block(
            x, filters=filters, stride=1,
            use_bn=use_bn, activation=activation,
            dropout_p=dropout_p, l2_weight=l2_weight,
            name=f"{name}_b{i}",
        )
    return x


def build_min_resnet(cfg: Dict[str, Any], num_classes: int) -> Model:
    """
    Minimal ResNet (explicit):
      stem -> stages -> GAP -> Dense(logits)

    Explicação do "stem" (arquitetura inicial):
    - Em ResNets maiores, o stem costuma ser algo como 7x7 stride 2 + maxpool.
    - Aqui vocês usam um stem minimalista: Conv 3x3 stride 1.
      Isso mantém resolução e reduz complexidade (bom para IMG_SIZE pequeno).

    GAP (GlobalAveragePooling2D):
    - Em vez de flattenar tudo (muitos parâmetros), faz média por canal.
    - Reduz overfitting e torna a rede mais "consciente" de presença de features.

    Dense(logits):
    - Saída em logits (sem softmax), pois a loss usa from_logits=True.
    """
    inp = layers.Input(
        shape=(cfg["IMG_SIZE"], cfg["IMG_SIZE"], cfg["IN_CHANNELS"]),
        name="input",
    )

    # STEM: primeira camada conv (feature extractor inicial)
    x = conv_bn_act(
        inp, filters=cfg["RES_CHANNELS"][0], k=3, stride=1,
        use_bn=cfg["USE_BN"], activation=cfg["ACTIVATION"],
        l2_weight=cfg["L2_WEIGHT"], name="stem",
    )

    # STAGES: empilhamento residual
    layers_per_stage = cfg["RES_LAYERS"]
    chs = cfg["RES_CHANNELS"]
    for s, (n_blocks, filters) in enumerate(zip(layers_per_stage, chs), start=1):
        # Estratégia padrão:
        # - estágio 1: stride 1 (sem downsample)
        # - estágios seguintes: stride 2 no primeiro bloco (downsample)
        stride = 1 if s == 1 else 2
        x = make_stage(
            x, filters=filters, n_blocks=n_blocks, first_stride=stride,
            use_bn=cfg["USE_BN"], activation=cfg["ACTIVATION"],
            dropout_p=cfg["BLOCK_DROPOUT"], l2_weight=cfg["L2_WEIGHT"],
            name=f"stage{s}",
        )

    # Pooling global (reduz HxW para 1 por canal)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Camada final: logits (dtype float32 para estabilidade, especialmente se mixed precision)
    logits = layers.Dense(
        num_classes,
        kernel_regularizer=regularizers.l2(cfg["L2_WEIGHT"]),
        dtype="float32",
        name="logits",
    )(x)

    return Model(inputs=inp, outputs=logits, name="ResNetSmallExplicit")


# =========================
# DATA LOADING + STRATIFIED KFOLD (NO SKLEARN)
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

    # Top classes por frequência (mantido)
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
    K-Fold estratificado (sem sklearn):
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
# TF.DATA PIPELINE (FAST)
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

    if training:
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
def make_optimizer(cfg: Dict[str, Any]):
    """
    Otimizador Adam (mantido).
    Se MIXED_PRECISION=True, encapsula com LossScaleOptimizer (mantido).
    """
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["LR"]))
    if bool(cfg.get("MIXED_PRECISION", False)):
        try:
            opt = mixed_precision.LossScaleOptimizer(opt)
        except Exception as e:
            print("[WARN] LossScaleOptimizer unavailable:", e)
    return opt


def make_loss():
    """
    SparseCategoricalCrossentropy:
    - rótulos inteiros (0..C-1)
    - from_logits=True porque o modelo retorna logits (sem softmax)
    """
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


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
    - forward: model(x, training=True) -> logits
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
            logits = model(x, training=True)

            # Loss base sempre em float32 (estabilidade numérica)
            base_loss = loss_fn(y, tf.cast(logits, tf.float32))
            base_loss = tf.cast(base_loss, tf.float32)

            # Regularização (L2) também em float32
            if model.losses:
                reg_loss = tf.add_n([tf.cast(l, tf.float32) for l in model.losses])
            else:
                reg_loss = tf.constant(0.0, dtype=tf.float32)

            loss = base_loss + reg_loss

        # Mixed precision loss scaling (mantido)
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) or (
                hasattr(optimizer, "get_scaled_loss") and hasattr(optimizer, "get_unscaled_gradients")
        ):
            scaled_loss = optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, model.trainable_variables)

        # Filtra gradientes None (mantido)
        grads_and_vars = [(g, v) for (g, v) in zip(grads, model.trainable_variables) if g is not None]
        if not grads_and_vars:
            raise RuntimeError(
                "Todos os gradientes vieram None. Possível overflow/NaN ou perda desconectada do grafo."
            )

        optimizer.apply_gradients(grads_and_vars)

        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
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

        # Debug do primeiro batch (muito útil para ver device/dtype)
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
        logits = model(x, training=False)

        base_loss = loss_fn(y, tf.cast(logits, tf.float32))
        base_loss = tf.cast(base_loss, tf.float32)

        if model.losses:
            reg_loss = tf.add_n([tf.cast(l, tf.float32) for l in model.losses])
        else:
            reg_loss = tf.constant(0.0, dtype=tf.float32)

        loss = base_loss + reg_loss

        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
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
# (3) PÓS-TREINO: "TESTE" + 10 EXEMPLOS + ONE-VS-ALL
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


def post_training_test_report(
    cfg: Dict[str, Any],
    run_dir: Path,
    df_val_as_test: pd.DataFrame,
    num_classes: int,
    label2idx: Dict[int, int],
    best_weights: List[np.ndarray],
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
    """
    print("\n" + "=" * 80)
    print("[POST] Avaliação final em 'teste' (val do melhor fold no K-Fold)")
    print("=" * 80)

    # Inverso para recuperar "label original" a partir do índice reindexado (y)
    idx2label = {v: k for k, v in label2idx.items()}

    # Dataset de teste (val do melhor fold), sem augment
    test_ds = make_tf_dataset(df_val_as_test, cfg, training=False)

    # Reconstrói e aplica pesos
    with tf.device(cfg["DEVICE"]):
        model = build_min_resnet(cfg, num_classes=num_classes)
    model.set_weights(best_weights)

    loss_fn = make_loss()
    te_loss, te_acc, te_err = evaluate(model, test_ds, loss_fn, cfg)

    print(f"[TEST] loss={te_loss:.6f} acc={te_acc:.6f} err={te_err:.6f}")
    print(f"[TEST] n={len(df_val_as_test)} classes={num_classes} device={cfg['DEVICE']}")

    # --------------------------------------------------------
    # 10 exemplos aleatórios de predição
    # --------------------------------------------------------
    n_show = min(10, len(df_val_as_test))
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

        # Predição (logits -> softmax para prob)
        logits = model(x, training=False)
        probs = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)[0].numpy()

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

    # Opcional: salvar relatórios no run_dir (não muda lógica do treino; só logging)
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
# K-FOLD CV RUNNER
# =========================
def run_kfold_cv(cfg: Dict[str, Any]) -> Tuple[float, Path]:
    """
    Executa K-Fold CV e grava arquivos de log.
    Retorna: (cv_mean_best_val_err, run_dir)

    Além disso (ajuste pedido):
    - Guarda o "melhor fold" (menor val_err) e faz um relatório final nele.
    """
    setup_tf_performance(cfg)
    set_seed(int(cfg["SEED"]))

    df, label2idx = load_and_filter_manifest(cfg)

    # Confirma resolução como ingerida (antes de resize)
    confirm_image_resolution(df, cfg)

    k = int(cfg["KFOLDS"])
    folds = stratified_kfold_indices(df, k=k, seed=int(cfg["SEED"]))
    num_classes = int(df["y"].nunique())

    stamp = now_stamp()
    run_dir = cfg["DATASET_DIR"] / cfg["RUNS_DIRNAME"] / stamp
    ensure_dir(run_dir)

    save_json(cfg, run_dir / "hyperparameters.json")
    save_json({str(k): int(v) for k, v in label2idx.items()}, run_dir / "label2idx.json")

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

        # Ajuste pedido: guardar o melhor fold global para "teste" final
        best_global_val_err = float("inf")
        best_global_weights: Optional[List[np.ndarray]] = None
        best_global_val_df: Optional[pd.DataFrame] = None
        best_global_fold_i: int = -1

        for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
            # Importante em CV: limpar grafo entre folds
            tf.keras.backend.clear_session()
            set_seed(int(cfg["SEED"]) + 1000 * fold_i)

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_ds = make_tf_dataset(train_df, cfg, training=True)
            val_ds = make_tf_dataset(val_df, cfg, training=False)

            # Cria modelo no device desejado
            with tf.device(cfg["DEVICE"]):
                model = build_min_resnet(cfg, num_classes=num_classes)

            optimizer = make_optimizer(cfg)
            loss_fn = make_loss()

            best_val_err = float("inf")
            best_epoch = -1
            best_weights = None

            print(f"\n[INFO] Fold {fold_i}/{k} | train={len(train_df)} val={len(val_df)} | classes={num_classes}")
            print(f"[INFO] Device scope: {cfg['DEVICE']}")

            for epoch in range(1, int(cfg["EPOCHS"]) + 1):
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
                    f"[Fold {fold_i}][Epoch {epoch}/{cfg['EPOCHS']}] "
                    f"train_loss={tr_loss:.4f} train_err={tr_err:.4f} | "
                    f"val_loss={va_loss:.4f} val_err={va_err:.4f} | {dt:.1f}s"
                )

                # Mantém melhor época do fold
                if va_err < best_val_err:
                    best_val_err = float(va_err)
                    best_epoch = int(epoch)
                    best_weights = model.get_weights()

            # Restaura pesos do melhor epoch do fold
            if best_weights is not None:
                model.set_weights(best_weights)

            best_val_errs.append(best_val_err)
            fold_summaries.append({
                "fold": fold_i,
                "best_epoch": best_epoch,
                "best_val_err": best_val_err,
            })

            # Guarda o melhor fold global para o relatório final (ajuste pedido)
            if best_val_err < best_global_val_err and best_weights is not None:
                best_global_val_err = float(best_val_err)
                best_global_weights = best_weights
                best_global_val_df = val_df.copy()
                best_global_fold_i = int(fold_i)

            # Salva modelo (opcional, mantido)
            if bool(cfg.get("SAVE_MODEL", False)):
                models_dir = run_dir / "models"
                ensure_dir(models_dir)
                save_path = models_dir / f"model_fold{fold_i}_best.keras"
                model.save(save_path)
                print(f"[SAVED] {save_path}")

    mean_err = float(np.mean(best_val_errs))
    std_err = float(np.std(best_val_errs, ddof=1)) if len(best_val_errs) > 1 else 0.0

    summary = {
        "kfolds": k,
        "num_images": int(len(df)),
        "num_classes": int(num_classes),
        "best_val_errs": best_val_errs,
        "cv_mean_best_val_err": mean_err,
        "cv_std_best_val_err": std_err,
        "folds": fold_summaries,
        "run_dir": str(run_dir),
        "device": cfg["DEVICE"],
        "timestamp": stamp,
        "mixed_precision": bool(cfg.get("MIXED_PRECISION", False)),
        "xla": bool(cfg.get("XLA", False)),
        "assume_ingested_size": bool(cfg.get("ASSUME_INGESTED_SIZE", False)),
    }
    save_json(summary, run_dir / "summary.json")

    print("\n[DONE] CV finished.")
    print(f"[RESULT] mean(best_val_err)={mean_err:.6f} | std={std_err:.6f}")
    print(f"[FILES] {run_dir}")

    # --------------------------------------------------------
    # Ajuste pedido (3): relatório final de "teste" + 10 exemplos + one-vs-all
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
            )
        else:
            print("[WARN] Could not run post-training report: best fold state not available.")
    except Exception as e:
        print("[WARN] Post-training report failed:", e)

    return mean_err, run_dir


# =========================
# MAIN
# =========================
def main() -> float:
    """
    Treina ResNet 'explicit' com K-Fold CV.
    Retorna o erro médio (val_err) dos melhores epochs por fold.
    """
    print("[INFO] Using device:", CONFIG["DEVICE"])
    mean_err, _run_dir = run_kfold_cv(CONFIG)
    return mean_err


if __name__ == "__main__":
    main()
