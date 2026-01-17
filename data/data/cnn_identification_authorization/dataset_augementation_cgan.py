# -*- coding: utf-8 -*-
"""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

cGAN (Conditional GAN) para *data augmentation* do dataset de faces (ex.: CelebA)
processado pelo seu pipeline de ingestão (manifest.csv + images/).

Referências (conceituais) usadas nos comentários:
- Goodfellow et al., 2014 (GAN)
- Mirza & Osindero, 2014 (cGAN)
- Radford et al., 2016 (DCGAN)
- Kingma & Ba, 2015 (Adam)
- Ioffe & Szegedy, 2015 (BatchNorm)
- Micikevicius et al., 2018 (mixed precision)

"""

from __future__ import annotations

import os
# -----------------------------------------------------------------------------
# DirectML: escolha do adaptador.
# Em máquinas com iGPU+dGPU, restringir a dGPU costuma reduzir instabilidades.
# Microsoft aponta o uso de DML_VISIBLE_DEVICES para selecionar adaptadores.
# (Documentação Microsoft Learn / TensorFlow-DirectML.)
# -----------------------------------------------------------------------------
os.environ.setdefault("DML_VISIBLE_DEVICES", "0")
# Reduz ruído de logs; mantenha em '0' se quiser tudo.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import math
import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import tensorflow as tf


# =============================================================================
# 1) CONFIG
# =============================================================================

# -----------------------------------------------------------------------------
# Paths robustos:
# A versão anterior usava paths relativos do tipo "data/...". Isso depende do
# *working directory* (cwd) do processo e, no seu caso, acabou virando
# ".../cnn_identification_authorization/data/..." e falhando.
#
# A ingestão usa um padrão "HERE = Path(__file__).parent" para tornar o script
# independente do cwd; aplicamos o mesmo aqui.
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE / "celeba_rgb_128x128"          # contém manifest.csv + images/
DEFAULT_OUTPUT_DIR = HERE / "cgan_outputs"               # outputs do treino + GAN_AUGMENTED_DATA


def _resolve_relative_to_here(p: Path) -> Path:
    """Resolve paths relativos em relação ao diretório do script (HERE)."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (HERE / p).resolve()

@dataclass(frozen=True)
class CganConfig:
    # ---------------------------
    # Paths
    # ---------------------------
    # Diretório vindo da ingestão: contém images/ e manifest.csv
    # Default: procura o dataset de ingestão na mesma pasta deste script.
    INPUT_DIR: Path = DEFAULT_INPUT_DIR

    # Diretório raiz de saída desta execução
    # Default: grava outputs na mesma pasta deste script.
    OUTPUT_DIR: Path = DEFAULT_OUTPUT_DIR

    # Subpasta exigida pelo pipeline downstream
    AUGMENTED_SUBDIR_NAME: str = "GAN_AUGMENTED_DATA"

    # ---------------------------
    # Dataset e imagens
    # ---------------------------
    # Tamanho usado no treino da GAN (menor -> muito mais rápido)
    IMG_SIZE_TRAIN: int = 64

    # Tamanho com que as imagens (originais + sintéticas) serão gravadas na pasta
    # GAN_AUGMENTED_DATA. Default 128 para ficar compatível com seu dataset ingestão.
    IMG_SIZE_SAVE: int = 128

    CHANNELS: int = 3

    # Quantidade máxima de imagens para treinar (debug). None/0 => usa tudo.
    MAX_TRAIN_IMAGES: int = 0

    # ---------------------------
    # Condição / classes
    # ---------------------------
    # Para 2k+ identidades, a cGAN fica difícil de aprender. Você pode limitar
    # quantas classes entram no treino, mantendo as mais frequentes.
    # 0 => usa todas.
    MAX_CLASSES_FOR_GAN: int = 0

    # ---------------------------
    # Hiperparâmetros GAN
    # ---------------------------
    Z_DIM: int = 100

    # Generator
    G_LABEL_EMBED_DIM: int = 100
    G_BASE_FILTERS: int = 256
    G_UPSAMPLE_BLOCKS: int = 3

    # Discriminator
    # (IMPORTANTE p/ eficiência): embedding pequeno + Dense p/ H*W.
    D_LABEL_EMBED_DIM: int = 128
    D_BASE_FILTERS: int = 64
    D_NUM_BLOCKS: int = 3

    # Treino
    BATCH_SIZE: int = 16
    EPOCHS: int = 3

    # Learning rate (padrões próximos aos usados em DCGAN)
    # Heurística popularizada em DCGAN: Adam com beta1=0.5 (Radford, Metz & Chintala, 2015).
    LR_G: float = 2e-4
    LR_D: float = 2e-4
    ADAM_BETA1: float = 0.5
    ADAM_BETA2: float = 0.999

    # Regularização leve no D para estabilidade (opcional)
    LABEL_SMOOTH_REAL: float = 0.9  # 1.0 desliga. Ideia comum em GANs.

    # ---------------------------
    # Performance / logs
    # ---------------------------
    SHUFFLE_BUFFER: int = 10_000
    PREFETCH: int = 2
    NUM_PARALLEL_CALLS: int = 4

    LOG_EVERY_STEPS: int = 100
    SAVE_SAMPLES_EVERY_EPOCH: int = 1
    SAMPLE_GRID_N: int = 36

    # Salvar checkpoints
    SAVE_CHECKPOINT_EVERY_EPOCH: int = 1

    # Mixed precision: em DirectML pode variar; mantenha False a menos que queira testar.
    USE_MIXED_PRECISION: bool = False

    # ---------------------------
    # Augmentation
    # ---------------------------
    # Multiplicador total do dataset final.
    # Ex.: 3x => dataset final ~3*N, ou seja, gera (3-1)*N imagens novas.
    AUGMENT_MULTIPLIER: float = 3.0

    # Batch para geração (pode ser maior que treino)
    GEN_BATCH_SIZE: int = 64

    # JPEG quality p/ salvar
    JPEG_QUALITY: int = 95


# Permite override sem editar o código:
#   set CGAN_INPUT_DIR=C:\...\celeba_rgb_128x128
#   set CGAN_OUTPUT_DIR=C:\...\cgan_outputs
CFG = CganConfig()
_env_in = os.environ.get("CGAN_INPUT_DIR")
_env_out = os.environ.get("CGAN_OUTPUT_DIR")
if _env_in:
    CFG = replace(CFG, INPUT_DIR=Path(_env_in))
if _env_out:
    CFG = replace(CFG, OUTPUT_DIR=Path(_env_out))


# =============================================================================
# 2) UTIL
# =============================================================================


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def maybe_enable_mixed_precision(enable: bool) -> None:
    """Mixed precision (Micikevicius et al., 2018). Em DirectML, suporte e ganho variam."""
    if not enable:
        return
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    print("[MP] Mixed precision policy:", mixed_precision.global_policy())


def configure_gpu_or_fail() -> None:
    """Garante que TF enxergue GPU (DirectML) e valida com um matmul simples."""
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    print("[DEVICES] CPUs:", cpus)
    print("[DEVICES] GPUs:", gpus)

    if not gpus:
        raise RuntimeError(
            "Nenhuma GPU foi detectada pelo TensorFlow. "
            "Verifique instalação do tensorflow-directml-plugin e drivers."
        )

    # Memory growth evita que TF tente reservar toda memória de cara.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # Smoke test: matmul em GPU:0
    with tf.device("/GPU:0"):
        a = tf.random.uniform((1024, 1024))
        b = tf.random.uniform((1024, 1024))
        c = tf.matmul(a, b)
        _ = c.numpy()  # força execução
        print("[GPU-TEST] matmul device:", c.device)


# =============================================================================
# 3) DATASET: ler manifest + mapear identidades -> índices contíguos
# =============================================================================


def load_ingestion_manifest(input_dir: Path) -> pd.DataFrame:
    """Lê manifest.csv do pipeline de ingestão.

    Espera:
      input_dir/
        images/
        manifest.csv

    Colunas mínimas:
      - image_name
      - label (identidade)

    Colunas opcionais: ok, dst, src, error
    """
    # Resolve paths relativos em relação ao diretório do script (HERE), não do cwd.
    # Isso evita o erro: ".../cnn_identification_authorization/data/...".
    input_dir = _resolve_relative_to_here(input_dir)

    manifest_path = input_dir / "manifest.csv"
    images_dir = input_dir / "images"

    if not manifest_path.exists() or not images_dir.exists():
        # Mensagem de erro mais útil: mostra o que foi tentado e sugere correção.
        tried = [str(manifest_path), str(images_dir)]
        msg = (
            "Não encontrei a saída do pipeline de ingestão no INPUT_DIR.\n"
            f"INPUT_DIR resolvido: {input_dir}\n"
            "Esperado:\n"
            f"  - {manifest_path}\n"
            f"  - {images_dir}\n\n"
            "Como corrigir:\n"
            "  1) Garanta que você rodou o script de ingestão e ele criou 'celeba_rgb_.../manifest.csv' e 'images/'.\n"
            "  2) Ajuste CFG.INPUT_DIR para apontar para a pasta que contém manifest.csv + images/.\n"
            "     Ex.: CFG = dataclasses.replace(CFG, INPUT_DIR=Path(r\"C:\\...\\celeba_rgb_128x128\"))\n"
        )
        raise FileNotFoundError(msg)

    df = pd.read_csv(manifest_path)

    if "image_name" not in df.columns or "label" not in df.columns:
        raise ValueError("manifest.csv precisa conter colunas: image_name, label")

    if "ok" in df.columns:
        df = df[df["ok"] == True].copy()

    df["label"] = df["label"].astype(int)
    df["image_name"] = df["image_name"].astype(str)

    # Resolve path
    if "dst" in df.columns:
        df["path"] = df["dst"].astype(str)
        df.loc[df["path"].isna() | (df["path"].str.len() == 0), "path"] = df["image_name"].apply(
            lambda n: str(images_dir / n)
        )
    else:
        df["path"] = df["image_name"].apply(lambda n: str(images_dir / n))

    # Filtra paths inexistentes
    df = df[df["path"].apply(lambda p: Path(p).exists())].copy()
    df = df.reset_index(drop=True)

    print(f"[DATA] Imagens carregadas do manifest: {len(df)} | identidades: {df['label'].nunique()}")
    return df


def maybe_limit_classes(df: pd.DataFrame, max_classes: int) -> pd.DataFrame:
    if max_classes and max_classes > 0:
        counts = df["label"].value_counts()
        keep = counts.head(max_classes).index.astype(int)
        df2 = df[df["label"].isin(keep)].copy().reset_index(drop=True)
        print(f"[DATA] Limitando para top-{max_classes} classes: {len(df2)} imgs | {df2['label'].nunique()} ids")
        return df2
    return df


def maybe_limit_images(df: pd.DataFrame, max_images: int, seed: int = 123) -> pd.DataFrame:
    if max_images and max_images > 0 and len(df) > max_images:
        df2 = df.sample(n=max_images, random_state=seed).copy().reset_index(drop=True)
        print(f"[DATA] Subamostrando para {max_images} imagens (debug)")
        return df2
    return df


def build_label_maps(labels: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Mapeia label original (ex.: 1001) -> índice [0..K-1] para Embedding."""
    uniq = np.unique(labels)
    label_to_index = {int(l): int(i) for i, l in enumerate(uniq)}
    index_to_label = {int(i): int(l) for i, l in enumerate(uniq)}
    return label_to_index, index_to_label


def preprocess_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    """Lê JPEG, converte para float32 em [-1, 1] e redimensiona.

    - Normalização [-1,1] combina naturalmente com saída tanh do gerador.
      (Goodfellow et al., 2014; Radford et al., 2016)
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=CFG.CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img = img * 2.0 - 1.0  # [-1,1]
    img.set_shape((img_size, img_size, CFG.CHANNELS))
    return img


def build_tf_dataset(
    paths: np.ndarray,
    label_indices: np.ndarray,
    batch_size: int,
    img_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Pipeline tf.data.

    Dica: usar `drop_remainder=True` mantém batch shape fixo, o que evita retrace e
    pode reduzir consumo de memória (importante em DirectML).
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, label_indices))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(CFG.SHUFFLE_BUFFER, len(paths)), reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = preprocess_image(p, img_size)
        y = tf.cast(y, tf.int32)
        y = tf.reshape(y, [1])  # (1,) por amostra -> (B,1) após batch
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=CFG.NUM_PARALLEL_CALLS)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(CFG.PREFETCH)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    return ds


# =============================================================================
# 4) MODELOS: Generator e Discriminator
# =============================================================================


def build_generator(num_classes: int) -> tf.keras.Model:
    """Generator condicional (cGAN): G(z, y) -> imagem.

    Estrutura inspirada em DCGAN (Radford et al., 2016) com condição via embedding
    (Mirza & Osindero, 2014).
    """
    noise_in = tf.keras.Input(shape=(CFG.Z_DIM,), name="noise")
    label_in = tf.keras.Input(shape=(1,), dtype="int32", name="label")

    # Embedding do rótulo
    label_emb = tf.keras.layers.Embedding(
        input_dim=num_classes,
        output_dim=CFG.G_LABEL_EMBED_DIM,
        name="label_emb",
    )(label_in)
    label_emb = tf.keras.layers.Flatten()(label_emb)

    x = tf.keras.layers.Concatenate()([noise_in, label_emb])

    # Começa pequeno e "upsample" por Conv2DTranspose
    start_res = CFG.IMG_SIZE_TRAIN // (2 ** CFG.G_UPSAMPLE_BLOCKS)  # ex: 64/(2^3)=8
    start_filters = CFG.G_BASE_FILTERS
    x = tf.keras.layers.Dense(start_res * start_res * start_filters, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)  # Ioffe & Szegedy, 2015
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((start_res, start_res, start_filters))(x)

    filters = start_filters
    for _ in range(CFG.G_UPSAMPLE_BLOCKS - 1):
        filters = max(filters // 2, 32)
        x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Saída: 3 canais, tanh
    x = tf.keras.layers.Conv2DTranspose(CFG.CHANNELS, 4, strides=2, padding="same", activation="tanh")(x)

    return tf.keras.Model([noise_in, label_in], x, name="generator")


def build_discriminator(num_classes: int) -> tf.keras.Model:
    """Discriminator condicional: D(x, y) -> logit (real vs fake).

    Condição via (embedding pequeno -> Dense -> mapa HxW), seguindo a ideia do slide
    de concatenar condição como canal extra, mas evitando Embedding direto para H*W
    (muito pesado em memória/compute).

    Isso preserva o mecanismo: concat([img, cond_map]) no input do D.
    """
    img_in = tf.keras.Input(shape=(CFG.IMG_SIZE_TRAIN, CFG.IMG_SIZE_TRAIN, CFG.CHANNELS), name="img")
    label_in = tf.keras.Input(shape=(1,), dtype="int32", name="label")

    # Embedding pequeno + Dense p/ H*W
    emb = tf.keras.layers.Embedding(num_classes, CFG.D_LABEL_EMBED_DIM, name="label_emb")(label_in)
    emb = tf.keras.layers.Flatten()(emb)
    emb = tf.keras.layers.Dense(CFG.IMG_SIZE_TRAIN * CFG.IMG_SIZE_TRAIN, use_bias=False)(emb)
    emb = tf.keras.layers.Reshape((CFG.IMG_SIZE_TRAIN, CFG.IMG_SIZE_TRAIN, 1))(emb)

    x = tf.keras.layers.Concatenate(axis=-1)([img_in, emb])

    filters = CFG.D_BASE_FILTERS
    for i in range(CFG.D_NUM_BLOCKS):
        x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        filters = min(filters * 2, 512)

    x = tf.keras.layers.Flatten()(x)
    # Logit (sem sigmoid) p/ usar BCE(from_logits=True) com mais estabilidade.
    x = tf.keras.layers.Dense(1, name="logit")(x)

    return tf.keras.Model([img_in, label_in], x, name="discriminator")


# =============================================================================
# 5) TRAIN LOOP (tf.function + GradientTape)
# =============================================================================


def make_train_steps(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    g_opt: tf.keras.optimizers.Optimizer,
    d_opt: tf.keras.optimizers.Optimizer,
):
    """Cria funções de treino compiladas.

    Usar GradientTape explícito é a forma mais robusta em GANs para evitar
    armadilhas de `trainable`/`compile` do Keras (é um ponto recorrente em discussões
    sobre Keras/TF quando se alterna treino de D e G).
    """

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def d_step(real_img: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch = tf.shape(real_img)[0]
        z = tf.random.normal((batch, CFG.Z_DIM))
        fake_img = generator([z, y], training=True)

        # Labels (opcional) com smoothing no real
        real_targets = tf.ones((batch, 1)) * tf.cast(CFG.LABEL_SMOOTH_REAL, tf.float32)
        fake_targets = tf.zeros((batch, 1))

        with tf.GradientTape() as tape:
            real_logits = discriminator([real_img, y], training=True)
            fake_logits = discriminator([fake_img, y], training=True)
            d_loss_real = bce(real_targets, real_logits)
            d_loss_fake = bce(fake_targets, fake_logits)
            d_loss = d_loss_real + d_loss_fake

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss, tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

    @tf.function
    def g_step(y: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(y)[0]
        z = tf.random.normal((batch, CFG.Z_DIM))
        targets = tf.ones((batch, 1))  # G quer que D(fake)=1

        with tf.GradientTape() as tape:
            fake_img = generator([z, y], training=True)
            fake_logits = discriminator([fake_img, y], training=False)
            g_loss = bce(targets, fake_logits)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss

    return d_step, g_step


# =============================================================================
# 6) AMOSTRAS / SAVE
# =============================================================================


def denorm_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converte [-1,1] -> uint8 [0,255]."""
    x = (img + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return x


def save_image_uint8(path: Path, arr_uint8: np.ndarray, quality: int = 95) -> None:
    from PIL import Image

    ensure_dir(path.parent)
    Image.fromarray(arr_uint8).save(str(path), format="JPEG", quality=int(quality), optimize=True)


def generate_and_save_samples(
    generator: tf.keras.Model,
    out_dir: Path,
    epoch: int,
    index_to_label: Dict[int, int],
    num_classes: int,
) -> None:
    """Gera um pequeno grid de amostras para inspeção visual."""
    ensure_dir(out_dir)

    n = CFG.SAMPLE_GRID_N
    # Escolhe algumas classes (índices contíguos) de forma determinística
    rng = np.random.default_rng(1234 + epoch)
    y_idx = rng.integers(0, num_classes, size=(n,), dtype=np.int32)
    y = y_idx.reshape(-1, 1)

    z = rng.normal(size=(n, CFG.Z_DIM)).astype(np.float32)

    with tf.device("/GPU:0"):
        fake = generator([z, y], training=False).numpy()

    fake_u8 = denorm_to_uint8(fake)

    # Salva como imagens individuais (mais simples/rápido que montar mosaico)
    for i in range(n):
        orig_label = index_to_label[int(y_idx[i])]
        p = out_dir / f"sample_e{epoch:03d}_{i:03d}_label{orig_label}.jpg"
        save_image_uint8(p, fake_u8[i], quality=CFG.JPEG_QUALITY)


# =============================================================================
# 7) AUGMENTAÇÃO: criar GAN_AUGMENTED_DATA com originais + sintéticas
# =============================================================================


def write_augmented_dataset(
    df_original: pd.DataFrame,
    generator: tf.keras.Model,
    label_to_index: Dict[int, int],
    out_root: Path,
) -> None:
    """Cria pasta auto-contida para o próximo treino consumir só mudando INPUT_DIR."""

    augmented_dir = out_root / CFG.AUGMENTED_SUBDIR_NAME
    images_dir = augmented_dir / "images"
    ensure_dir(images_dir)

    # Copia as imagens originais
    print("[AUG] Copiando imagens originais...")
    rows: List[Dict] = []

    for i, row in enumerate(df_original.itertuples(index=False), start=1):
        src_path = Path(row.path)
        dst_path = images_dir / row.image_name
        try:
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            rows.append({
                "image_name": row.image_name,
                "label": int(row.label),
                "src": str(src_path),
                "dst": str(dst_path),
                "ok": True,
                "error": "",
                "is_synthetic": False,
            })
        except Exception as e:
            rows.append({
                "image_name": row.image_name,
                "label": int(row.label),
                "src": str(src_path),
                "dst": str(dst_path),
                "ok": False,
                "error": str(e),
                "is_synthetic": False,
            })

        if i % 5000 == 0:
            print(f"[AUG] copiados: {i}/{len(df_original)}")

    # Quantas imagens novas gerar?
    n_orig_ok = int(sum(1 for r in rows if r["ok"]))
    mult = float(CFG.AUGMENT_MULTIPLIER)
    if mult < 1.0:
        raise ValueError("AUGMENT_MULTIPLIER deve ser >= 1.0")

    n_new = int(round((mult - 1.0) * n_orig_ok))
    print(f"[AUG] AUGMENT_MULTIPLIER={mult} => gerar ~{n_new} imagens sintéticas")

    if n_new <= 0:
        manifest = pd.DataFrame(rows)
        manifest.to_csv(augmented_dir / "manifest.csv", index=False)
        print("[AUG] Nenhuma imagem sintética solicitada. Manifest salvo.")
        return

    # Distribuição por label: escala por (mult-1)
    df_ok = df_original.copy()
    if "ok" in df_ok.columns:
        df_ok = df_ok[df_ok["ok"] == True].copy()

    label_counts = df_ok["label"].value_counts().to_dict()

    # Para evitar drift, geramos por label seguindo a contagem original.
    to_generate: List[int] = []
    for lbl, cnt in label_counts.items():
        k = int(round((mult - 1.0) * cnt))
        if k > 0:
            to_generate.extend([int(lbl)] * k)

    # Ajuste fino para bater n_new
    if len(to_generate) > n_new:
        to_generate = to_generate[:n_new]
    elif len(to_generate) < n_new:
        # completa com labels mais frequentes
        top_labels = [int(k) for k, _ in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)]
        j = 0
        while len(to_generate) < n_new:
            to_generate.append(top_labels[j % len(top_labels)])
            j += 1

    assert len(to_generate) == n_new

    print("[AUG] Gerando imagens sintéticas...")

    rng = np.random.default_rng(2026)
    gen_batch = int(CFG.GEN_BATCH_SIZE)

    # Nomes únicos
    stamp = utc_now_compact()

    produced = 0
    batch_id = 0

    while produced < n_new:
        this = min(gen_batch, n_new - produced)
        labels_orig = np.array(to_generate[produced:produced + this], dtype=np.int32)

        # Converte label original -> índice contíguo
        labels_idx = np.array([label_to_index[int(l)] for l in labels_orig], dtype=np.int32).reshape(-1, 1)
        z = rng.normal(size=(this, CFG.Z_DIM)).astype(np.float32)

        with tf.device("/GPU:0"):
            fake = generator([z, labels_idx], training=False).numpy()

        # Opcional: salvar no tamanho do dataset original
        if CFG.IMG_SIZE_SAVE != CFG.IMG_SIZE_TRAIN:
            # Resize em TF (CPU/GPU). Mantém simples.
            fake_tf = tf.convert_to_tensor(fake, dtype=tf.float32)
            fake_tf = tf.image.resize(fake_tf, (CFG.IMG_SIZE_SAVE, CFG.IMG_SIZE_SAVE), method="bilinear")
            fake = fake_tf.numpy()

        fake_u8 = denorm_to_uint8(fake)

        for i in range(this):
            lbl = int(labels_orig[i])
            img_name = f"gan_{stamp}_b{batch_id:05d}_{i:03d}_label{lbl}.jpg"
            dst_path = images_dir / img_name
            try:
                save_image_uint8(dst_path, fake_u8[i], quality=CFG.JPEG_QUALITY)
                rows.append({
                    "image_name": img_name,
                    "label": lbl,
                    "src": "GAN",
                    "dst": str(dst_path),
                    "ok": True,
                    "error": "",
                    "is_synthetic": True,
                })
            except Exception as e:
                rows.append({
                    "image_name": img_name,
                    "label": lbl,
                    "src": "GAN",
                    "dst": str(dst_path),
                    "ok": False,
                    "error": str(e),
                    "is_synthetic": True,
                })

        produced += this
        batch_id += 1

        if produced % 5000 == 0 or produced == n_new:
            print(f"[AUG] sintéticas geradas: {produced}/{n_new}")

    manifest = pd.DataFrame(rows)
    manifest_path = augmented_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Metadados auxiliares
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(CFG.INPUT_DIR),
        "img_size_train": CFG.IMG_SIZE_TRAIN,
        "img_size_save": CFG.IMG_SIZE_SAVE,
        "augment_multiplier": CFG.AUGMENT_MULTIPLIER,
        "n_original": n_orig_ok,
        "n_synthetic_requested": n_new,
        "n_total_rows": int(len(manifest)),
        "dml_visible_devices": os.environ.get("DML_VISIBLE_DEVICES", ""),
    }
    (augmented_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[AUG] Dataset aumentado salvo em:", augmented_dir)
    print("[AUG] Imagens em:", images_dir)
    print("[AUG] Manifest:", manifest_path)


# =============================================================================
# 8) MAIN
# =============================================================================


def main() -> None:
    maybe_enable_mixed_precision(CFG.USE_MIXED_PRECISION)
    configure_gpu_or_fail()

    # OUTPUT_DIR pode ser relativo; resolvemos em relação ao diretório do script (HERE)
    # para não depender do cwd do processo.
    output_root = _resolve_relative_to_here(CFG.OUTPUT_DIR)
    run_dir = output_root / f"run_{utc_now_compact()}"
    ensure_dir(run_dir)

    # Logs de amostras
    samples_dir = run_dir / "samples"
    ensure_dir(samples_dir)

    # 1) Carrega dados
    df = load_ingestion_manifest(CFG.INPUT_DIR)
    df = maybe_limit_classes(df, CFG.MAX_CLASSES_FOR_GAN)
    df = maybe_limit_images(df, CFG.MAX_TRAIN_IMAGES)

    labels = df["label"].values.astype(np.int32)
    label_to_index, index_to_label = build_label_maps(labels)

    y_idx = np.array([label_to_index[int(l)] for l in labels], dtype=np.int32)
    paths = df["path"].values.astype(str)

    num_classes = int(len(label_to_index))
    print(f"[DATA] num_classes (contíguo) = {num_classes}")

    # 2) Dataset
    ds = build_tf_dataset(
        paths=paths,
        label_indices=y_idx,
        batch_size=CFG.BATCH_SIZE,
        img_size=CFG.IMG_SIZE_TRAIN,
        shuffle=True,
    )

    steps_per_epoch = int(len(paths) // CFG.BATCH_SIZE)
    print(f"[DATA] steps_per_epoch (drop_remainder) = {steps_per_epoch}")

    # 3) Modelos
    generator = build_generator(num_classes)
    discriminator = build_discriminator(num_classes)

    print("[MODEL] Generator summary:")
    generator.summary()
    print("[MODEL] Discriminator summary:")
    discriminator.summary()

    # 4) Otimizadores
    g_opt = tf.keras.optimizers.Adam(learning_rate=CFG.LR_G, beta_1=CFG.ADAM_BETA1, beta_2=CFG.ADAM_BETA2)
    d_opt = tf.keras.optimizers.Adam(learning_rate=CFG.LR_D, beta_1=CFG.ADAM_BETA1, beta_2=CFG.ADAM_BETA2)

    d_step, g_step = make_train_steps(generator, discriminator, g_opt, d_opt)

    # 5) Checkpoint
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        g_opt=g_opt,
        d_opt=d_opt,
    )
    ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=str(run_dir / "checkpoints"), max_to_keep=3)

    # 6) Treino
    print("[GAN] Iniciando treino...")

    history_rows = []

    for epoch in range(1, CFG.EPOCHS + 1):
        d_loss_m = tf.keras.metrics.Mean()
        g_loss_m = tf.keras.metrics.Mean()
        d_real_m = tf.keras.metrics.Mean()
        d_fake_m = tf.keras.metrics.Mean()

        for step, (real_img, y) in enumerate(ds, start=1):
            d_loss, d_real_logit, d_fake_logit = d_step(real_img, y)
            g_loss = g_step(y)

            d_loss_m.update_state(d_loss)
            g_loss_m.update_state(g_loss)
            d_real_m.update_state(d_real_logit)
            d_fake_m.update_state(d_fake_logit)

            if step % CFG.LOG_EVERY_STEPS == 0 or step == 1:
                print(
                    f"[E{epoch:03d} S{step:05d}/{steps_per_epoch}] "
                    f"D={d_loss_m.result().numpy():.4f} "
                    f"G={g_loss_m.result().numpy():.4f} "
                    f"D(real_logit)={d_real_m.result().numpy():.3f} "
                    f"D(fake_logit)={d_fake_m.result().numpy():.3f}"
                )

            # Para evitar loop infinito caso ds seja maior que esperado (por segurança)
            if step >= steps_per_epoch:
                break

        row = {
            "epoch": epoch,
            "d_loss": float(d_loss_m.result().numpy()),
            "g_loss": float(g_loss_m.result().numpy()),
            "d_real_logit_mean": float(d_real_m.result().numpy()),
            "d_fake_logit_mean": float(d_fake_m.result().numpy()),
        }
        history_rows.append(row)

        print(f"[E{epoch:03d}] DONE | D={row['d_loss']:.4f} | G={row['g_loss']:.4f}")

        # Salvar amostras
        if CFG.SAVE_SAMPLES_EVERY_EPOCH and (epoch % CFG.SAVE_SAMPLES_EVERY_EPOCH == 0):
            print("[SAMPLES] Gerando amostras...")
            generate_and_save_samples(generator, samples_dir, epoch, index_to_label, num_classes)

        # Checkpoint
        if CFG.SAVE_CHECKPOINT_EVERY_EPOCH and (epoch % CFG.SAVE_CHECKPOINT_EVERY_EPOCH == 0):
            p = ckpt_mgr.save(checkpoint_number=epoch)
            print("[CKPT] salvo em:", p)

        # Log CSV
        pd.DataFrame(history_rows).to_csv(run_dir / "training_log.csv", index=False)

    # 7) Salva gerador final
    gen_path = run_dir / "generator.keras"
    generator.save(gen_path)
    print("[SAVE] Generator salvo em:", gen_path)

    # 8) Cria dataset aumentado
    write_augmented_dataset(df_original=df, generator=generator, label_to_index=label_to_index, out_root=run_dir)

    print("[DONE] run_dir:", run_dir)


if __name__ == "__main__":
    main()


# =============================================================================
# NOTAS IMPORTANTES (Windows + DirectML)
# =============================================================================
#
# 1) DXGI_ERROR_DEVICE_HUNG:
#    Esse erro normalmente significa timeout (TDR) do driver ao rodar kernels longos.
#    Microsoft recomenda ajustar TdrDelay/TdrDdiDelay se for um fluxo de treino pesado.
#    Veja: aka.ms/tfdmltimeout e documentação do TensorFlow-DirectML.
#
# 2) Seleção de GPU:
#    Se sua máquina tem iGPU + dGPU, usar a dGPU é recomendável. Você pode controlar
#    isso com DML_VISIBLE_DEVICES (por exemplo, "0" ou "1").
#
# 3) Referências acadêmicas (base):
#    - Goodfellow, I. et al. (2014). Generative Adversarial Nets.
#    - Mirza, M.; Osindero, S. (2014). Conditional GANs.
#    - Radford, A.; Metz, L.; Chintala, S. (2016). Unsupervised Representation
#      Learning with Deep Convolutional GANs (DCGAN).
#    - Kingma, D. P.; Ba, J. (2015). Adam: A Method for Stochastic Optimization.
#    - Ioffe, S.; Szegedy, C. (2015). Batch Normalization.
#    - Micikevicius, P. et al. (2018). Mixed Precision Training.
