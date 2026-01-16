# -*- coding: utf-8 -*-
"""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Treinamento de cGAN (Conditional GAN) para Augmentation de Faces (CelebA)
========================================================================

Este script:
1) Lê imagens já processadas pelo pipeline de ingestão:
   - INPUT_DATA_DIR/images/*.jpg
   - INPUT_DATA_DIR/manifest.csv  (colunas mínimas: image_name, label; opcional: ok, dst)
2) Treina uma cGAN (G(z,y)->x e D(x,y)->{0,1}):
   - Gerador: noise + embedding(label) -> Dense -> Conv2DTranspose ... -> tanh
   - Discriminador: concat(img, embedding(label)->(H,W,1)) -> Conv2D ... -> sigmoid
   - Treino alternado: atualiza D com real/fake e depois G para "enganar" D
3) Após o treino, cria OUTPUT_DIR/GAN_AUGMENTED_DATA com:
   - images/ : cópia das imagens originais + imagens sintéticas geradas
   - manifest.csv : metadados completos preservando a identidade original (label)

GPU / DirectML:
- Em Windows usa "tensorflow-directml" (ou TF + plugin DirectML).
- O script valida presença de device GPU e roda um matmul teste imprimindo o device.
- Se REQUIRE_GPU=True e não houver GPU, o script falha explicitamente.

Referências acadêmicas (por etapa) estão no final e também em comentários localizados:
- GAN: Goodfellow et al., 2014
- cGAN: Mirza & Osindero, 2014
- DCGAN-style conv stacks: Radford et al., 2016
- Adam: Kingma & Ba, 2015
- BatchNorm: Ioffe & Szegedy, 2015
- ReLU: Nair & Hinton, 2010
- LeakyReLU: Maas et al., 2013
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf


# =============================================================================
# 1) CONFIGURAÇÕES
# =============================================================================

@dataclass
class CganConfig:
    # -------------------------
    # Paths (entrada / saída)
    # -------------------------
    # Aponte para o OUTPUT_DIR da ingestão (onde existem "images/" e "manifest.csv")
    # Ex.: <project_root>/data/cnn_identification_authorization/celeba_rgb_128x128
    INPUT_DATA_DIR: str = r"data/cnn_identification_authorization/celeba_rgb_128x128"

    # Diretório onde colocaremos GAN_AUGMENTED_DATA (subpasta dentro do INPUT_DATA_DIR por padrão)
    OUTPUT_ROOT_DIR: str = ""  # se vazio, usa INPUT_DATA_DIR

    # -------------------------
    # cGAN: parâmetros centrais
    # -------------------------
    IMG_SIZE: int = 64              # Alinha com a implementação dos slides (D espera 64x64).
    CHANNELS: int = 3               # RGB
    LATENT_DIM: int = 100           # slides sugerem 100
    EMBED_DIM: int = 100            # pode ser = LATENT_DIM (embedding do rótulo)
    UPSAMPLE_BLOCKS: int = 3        # 8->16->32->64 (3 upsample) para IMG_SIZE=64

    # Treino
    EPOCHS: int = 30
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 2e-4     # Adam(0.0002, 0.5)
    ADAM_BETA1: float = 0.5
    BUFFER_SIZE: int = 10_000
    PRINT_EVERY_STEPS: int = 100    # log a cada N batches
    SAVE_SAMPLES_EVERY_EPOCH: int = 1  # salva grid de amostras a cada N épocas

    # -------------------------
    # Dataset sampling (opcional)
    # -------------------------
    # cGAN com subset (para testar rápido), ajustar:
    MAX_IMAGES_FOR_GAN: int | None = None   # ex.: 20_000 ou None (usa tudo)
    MAX_IDENTITIES_FOR_GAN: int | None = None  # ex.: 200 ou None (usa todas)

    # -------------------------
    # Augmentation: multiplicador de imagens
    # -------------------------
    # MULTIPLIER=3 => dataset final terá ~3x o tamanho do original:
    # gera (MULTIPLIER-1)*N imagens novas e copia N originais => total ~ MULTIPLIER*N
    MULTIPLIER: int = 3

    # Controle de geração
    GEN_JPEG_QUALITY: int = 95
    GEN_PREFIX: str = "gan"

    # -------------------------
    # GPU / DirectML
    # -------------------------
    REQUIRE_GPU: bool = True        # se True, falha se não detectar GPU
    ENABLE_MIXED_PRECISION: bool = False  # opcional (pode ajudar; teste com cuidado)

    # Reprodutibilidade parcial
    RANDOM_SEED: int = 42


CFG = CganConfig()


# =============================================================================
# 2) UTIL: GPU (DirectML) + Reprodutibilidade + IO
# =============================================================================

def set_seeds(seed: int) -> None:
    """Define seeds para reduzir variância (não garante determinismo total em GPU)."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_or_fail(require_gpu: bool) -> None:
    """
    Valida GPU e imprime diagnóstico.

    Observação:
    - No tensorflow-directml, frequentemente aparece como device_type='GPU'.
    - O teste mais robusto aqui é:
        (a) list_physical_devices('GPU')
        (b) executar uma operação e imprimir tensor.device
    """
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    print("[DEVICES] CPUs:", cpus)
    print("[DEVICES] GPUs:", gpus)

    if require_gpu and not gpus:
        raise RuntimeError(
            "Nenhuma GPU detectada via TensorFlow. "
            "Se você está no Windows/DirectML, verifique se está usando tensorflow-directml "
            "ou se o backend DirectML está corretamente instalado."
        )

    # Evita que o TF tente alocar toda a VRAM de uma vez (quando suportado).
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # Teste simples: matmul e impressão do device onde o tensor foi materializado.
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    c = tf.matmul(a, b)
    # Força materialização
    _ = c.numpy()
    print("[GPU-TEST] matmul device:", c.device)


def maybe_enable_mixed_precision(enable: bool) -> None:
    """
    Mixed precision (FP16) pode acelerar em GPUs modernas, mas em DirectML pode variar.
    Use com cautela e compare estabilidade/qualidade.
    Ref: Micikevicius et al., 2018 (mixed precision training).
    """
    if not enable:
        return
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    print("[MP] Mixed precision policy:", mixed_precision.global_policy())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# 3) DATASET: ler manifest + mapear identidades -> índices contíguos (para Embedding)
# =============================================================================

def load_ingestion_manifest(input_dir: Path) -> pd.DataFrame:
    """
    Espera estrutura do pipeline de ingestão:
      input_dir/
        images/
        manifest.csv

    O manifest original (da ingestão) tem pelo menos:
      - image_name
      - label (identidade numérica)
    e opcionalmente:
      - ok (bool)
      - dst (path absoluto)
    """
    manifest_path = input_dir / "manifest.csv"
    images_dir = input_dir / "images"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Não encontrei manifest.csv em: {manifest_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Não encontrei pasta images/ em: {images_dir}")

    df = pd.read_csv(manifest_path)

    if "image_name" not in df.columns or "label" not in df.columns:
        raise ValueError("manifest.csv precisa conter colunas: image_name, label")

    # Se existir 'ok', mantemos somente as imagens ok=True
    if "ok" in df.columns:
        df = df[df["ok"] == True].copy()

    df["label"] = df["label"].astype(int)
    df["image_name"] = df["image_name"].astype(str)

    # Resolve path do arquivo:
    # - se existir 'dst' e for válido, usa; senão, usa images_dir/image_name
    if "dst" in df.columns:
        df["path"] = df["dst"].astype(str)
        # Alguns manifests podem ter dst vazio; fallback
        df.loc[df["path"].isna() | (df["path"].str.len() == 0), "path"] = df["image_name"].apply(
            lambda n: str(images_dir / n)
        )
    else:
        df["path"] = df["image_name"].apply(lambda n: str(images_dir / n))

    # Garante existência
    missing = [p for p in df["path"].tolist() if not Path(p).exists()]
    if missing:
        print("[WARN] Alguns paths do manifest não existem no disco. Exemplo:", missing[:5])
        df = df[df["path"].apply(lambda p: Path(p).exists())].copy()

    df = df.reset_index(drop=True)
    print(f"[DATA] Imagens carregadas do manifest: {len(df)} | identidades: {df['label'].nunique()}")
    return df


def make_label_mapping(df: pd.DataFrame, max_identities: int | None) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Para Embedding em cGAN, é muito mais eficiente ter classes 0..K-1 contíguas.
    Então criamos:
      - identity_id (original) -> class_index (contíguo)
    e preservamos identity_id no manifest final.

    Se max_identities for definido, pegamos as identidades mais frequentes.
    """
    counts = df["label"].value_counts()
    if max_identities is not None:
        top_ids = counts.nlargest(max_identities).index.tolist()
        df = df[df["label"].isin(top_ids)].copy()
        df = df.reset_index(drop=True)
        counts = df["label"].value_counts()
        print(f"[DATA] Filtrando para TOP {max_identities} identidades (mais frequentes).")

    identity_ids = counts.index.tolist()
    id_to_class = {int(identity_id): int(i) for i, identity_id in enumerate(identity_ids)}
    class_to_id = {v: k for k, v in id_to_class.items()}

    df["class_index"] = df["label"].map(id_to_class).astype(int)

    print(f"[DATA] num_classes (contíguo) = {len(id_to_class)}")
    return df, id_to_class, class_to_id


def maybe_subsample_images(df: pd.DataFrame, max_images: int | None, seed: int) -> pd.DataFrame:
    """Subamostra imagens para testes rápidos."""
    if max_images is None or max_images >= len(df):
        return df
    df2 = df.sample(n=max_images, random_state=seed).reset_index(drop=True)
    print(f"[DATA] Subamostra de imagens: {len(df2)}/{len(df)}")
    return df2


def tf_parse_image(path: tf.Tensor, label: tf.Tensor, img_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Lê JPEG do disco -> float32 -> resize -> normaliza para [-1, 1] (compatível com tanh no gerador).

    Normalização [-1,1] é prática comum em GANs/DCGAN (Radford et al., 2016),
    combinando bem com saída tanh do gerador.
    """
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    img = (img * 2.0) - 1.0  # [0,1] -> [-1,1]
    label = tf.cast(label, tf.int32)
    # Modelos Keras do slide usam Input(shape=(1,)); garantimos (batch,1)
    label = tf.reshape(label, [1])
    return img, label


def build_tf_dataset(df: pd.DataFrame, img_size: int, batch_size: int, buffer_size: int) -> tf.data.Dataset:
    paths = df["path"].tolist()
    labels = df["class_index"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=min(buffer_size, len(df)), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: tf_parse_image(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# 4) MODELOS: Gerador e Discriminador (conforme slides)
# =============================================================================

def build_generator(latent_dim: int, num_classes: int, img_size: int, channels: int,
                    embed_dim: int, upsample_blocks: int) -> tf.keras.Model:
    """
    Gerador cGAN conforme slides:
      noise z + embedding(label) -> concat -> Dense -> BN -> ReLU -> reshape -> Conv2DTranspose blocks -> tanh

    Referências:
    - cGAN: Mirza & Osindero, 2014
    - DCGAN (conv transpose + BN + ReLU + tanh): Radford et al., 2016
    - BatchNorm: Ioffe & Szegedy, 2015
    - ReLU: Nair & Hinton, 2010
    """
    assert img_size % (2 ** upsample_blocks) == 0, "IMG_SIZE precisa ser múltiplo de 2**UPSAMPLE_BLOCKS."
    start_res = img_size // (2 ** upsample_blocks)  # ex.: 64//8=8

    noise = tf.keras.layers.Input(shape=(latent_dim,), name="noise")
    label = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label")

    # Embedding do label para vetor (como no slide: Embedding(num_classes, latent_dim))
    label_embedding = tf.keras.layers.Embedding(num_classes, embed_dim, name="label_emb")(label)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)

    x = tf.keras.layers.Concatenate()([noise, label_embedding])

    # Dense para mapa espacial inicial
    x = tf.keras.layers.Dense(start_res * start_res * 256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape((start_res, start_res, 256))(x)

    # Blocos de upsample (Conv2DTranspose stride 2)
    # No slide: 128, depois 64, depois saída 3 canais com tanh (para 64x64)
    filters = [128, 64]
    for i in range(upsample_blocks - 1):
        f = filters[i] if i < len(filters) else max(32, 128 // (2 ** i))
        x = tf.keras.layers.Conv2DTranspose(
            f, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # Último upsample -> imagem final
    img = tf.keras.layers.Conv2DTranspose(
        channels, kernel_size=4, strides=2, padding="same", activation="tanh"
    )(x)

    return tf.keras.Model([noise, label], img, name="generator")


def build_discriminator(num_classes: int, img_size: int, channels: int) -> tf.keras.Model:
    """
    Discriminador cGAN conforme slides:
      img + embedding(label)->(H,W,1) concatenados no canal -> conv blocks -> sigmoid

    Referências:
    - cGAN: Mirza & Osindero, 2014
    - DCGAN discriminator style (Conv2D + LeakyReLU + sigmoid): Radford et al., 2016
    - LeakyReLU: Maas et al., 2013
    """
    img = tf.keras.layers.Input(shape=(img_size, img_size, channels), name="img")
    label = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label")

    # No slide: Embedding(num_classes, 64*64) e reshape para (64,64,1).
    # Aqui generalizamos para (H*W) -> (H,W,1).
    label_embedding = tf.keras.layers.Embedding(num_classes, img_size * img_size, name="label_emb")(label)
    label_embedding = tf.keras.layers.Reshape((img_size, img_size, 1))(label_embedding)

    x = tf.keras.layers.Concatenate(axis=-1)([img, label_embedding])

    x = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model([img, label], out, name="discriminator")


# =============================================================================
# 5) TREINAMENTO: loop alternado D/G (conforme slides)
# =============================================================================

def train_cgan(
    ds: tf.data.Dataset,
    num_classes: int,
    out_dir: Path,
    cfg: CganConfig,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Loop de treino semelhante ao do slide:
      for epoch:
        for real_imgs, labels:
          valid=1, fake=0
          fake_imgs = G(noise, labels)
          treina D em real (valid) e fake (fake)
          sampled_labels aleatórios
          treina cGAN (G com D congelado) para produzir valid

    Referências:
    - GAN objective/training: Goodfellow et al., 2014
    - Algoritmo de treino em minibatch: também refletido no pseudocódigo do slide
    - Adam: Kingma & Ba, 2015
    """
    ensure_dir(out_dir)
    samples_dir = out_dir / "cgan_samples"
    ensure_dir(samples_dir)
    models_dir = out_dir / "cgan_models"
    ensure_dir(models_dir)

    generator = build_generator(
        latent_dim=cfg.LATENT_DIM,
        num_classes=num_classes,
        img_size=cfg.IMG_SIZE,
        channels=cfg.CHANNELS,
        embed_dim=cfg.EMBED_DIM,
        upsample_blocks=cfg.UPSAMPLE_BLOCKS,
    )
    discriminator = build_discriminator(
        num_classes=num_classes,
        img_size=cfg.IMG_SIZE,
        channels=cfg.CHANNELS,
    )

    # Discriminador: BCE + Adam(2e-4, beta1=0.5) (como no slide)
    d_opt = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE, beta_1=cfg.ADAM_BETA1)
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=d_opt,
        metrics=["accuracy"],
    )

    # cGAN: D congelado, treina-se G via loss do D
    discriminator.trainable = False

    z = tf.keras.layers.Input(shape=(cfg.LATENT_DIM,), name="noise_in")
    y = tf.keras.layers.Input(shape=(1,), dtype="int32", name="label_in")
    fake_img = generator([z, y])
    validity = discriminator([fake_img, y])

    cgan = tf.keras.Model([z, y], validity, name="cgan")
    g_opt = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE, beta_1=cfg.ADAM_BETA1)
    cgan.compile(loss="binary_crossentropy", optimizer=g_opt)

    print("[MODEL] Generator summary:")
    generator.summary()
    print("[MODEL] Discriminator summary:")
    discriminator.summary()

    step = 0
    t0 = time.time()

    # Para gerar amostras consistentes ao longo do treino
    fixed_noise = tf.random.normal((16, cfg.LATENT_DIM))
    fixed_labels = tf.random.uniform((16,), minval=0, maxval=num_classes, dtype=tf.int32)
    fixed_labels = tf.reshape(fixed_labels, (-1, 1))

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_t0 = time.time()

        # métricas simples agregadas por época
        d_loss_vals = []
        g_loss_vals = []

        for real_imgs, labels in ds:
            step += 1
            batch = tf.shape(real_imgs)[0]

            valid = tf.ones((batch, 1), dtype=tf.float32)
            fake = tf.zeros((batch, 1), dtype=tf.float32)

            # -----------------------
            # Treina Discriminador
            # -----------------------
            noise = tf.random.normal((batch, cfg.LATENT_DIM))
            fake_imgs = generator([noise, labels], training=False)

            # D em real
            d_loss_real = discriminator.train_on_batch([real_imgs, labels], valid)
            # D em fake
            d_loss_fake = discriminator.train_on_batch([fake_imgs, labels], fake)

            # d_loss_real/fake retornam [loss, acc]
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])

            # -----------------------
            # Treina Gerador (via cGAN)
            # -----------------------
            noise2 = tf.random.normal((batch, cfg.LATENT_DIM))
            sampled_labels = tf.random.uniform((batch,), minval=0, maxval=num_classes, dtype=tf.int32)
            sampled_labels = tf.reshape(sampled_labels, (-1, 1))

            g_loss = cgan.train_on_batch([noise2, sampled_labels], valid)

            d_loss_vals.append(float(d_loss))
            g_loss_vals.append(float(g_loss))

            # -----------------------
            # Logs / contadores
            # -----------------------
            if step % cfg.PRINT_EVERY_STEPS == 0:
                elapsed = time.time() - t0
                print(
                    f"[TRAIN] epoch={epoch:03d}/{cfg.EPOCHS} "
                    f"step={step:07d} "
                    f"D_loss={d_loss:.4f} D_acc={d_acc:.4f} "
                    f"G_loss={float(g_loss):.4f} "
                    f"elapsed={elapsed/60:.1f}min"
                )

        # Salva amostras a cada N épocas (grid 4x4)
        if epoch % cfg.SAVE_SAMPLES_EVERY_EPOCH == 0:
            gen_imgs = generator([fixed_noise, fixed_labels], training=False)
            save_image_grid(
                gen_imgs,
                fixed_labels,
                out_path=samples_dir / f"samples_epoch_{epoch:03d}.jpg",
                nrow=4,
            )

        epoch_elapsed = time.time() - epoch_t0
        print(
            f"[EPOCH-END] epoch={epoch:03d} "
            f"mean_D_loss={np.mean(d_loss_vals):.4f} "
            f"mean_G_loss={np.mean(g_loss_vals):.4f} "
            f"time={epoch_elapsed:.1f}s"
        )

    # Salva modelos
    gen_path = models_dir / "generator.keras"
    disc_path = models_dir / "discriminator.keras"
    generator.save(gen_path)
    # discriminator está com trainable=False no grafo do cgan; recriamos trainable para salvar ok
    discriminator.trainable = True
    discriminator.save(disc_path)

    print("[SAVED] Generator:", gen_path)
    print("[SAVED] Discriminator:", disc_path)

    return generator, discriminator


def save_image_grid(images: tf.Tensor, labels: tf.Tensor, out_path: Path, nrow: int = 4) -> None:
    """
    Salva um grid simples de imagens (nrow x nrow).

    images estão em [-1,1] (tanh). Reescala para [0,255] e salva JPEG.
    """
    ensure_dir(out_path.parent)

    imgs = images.numpy()
    imgs = (imgs + 1.0) / 2.0  # [-1,1] -> [0,1]
    imgs = np.clip(imgs * 255.0, 0, 255).astype(np.uint8)

    # Constrói grid
    n = imgs.shape[0]
    ncol = nrow
    nlin = int(math.ceil(n / ncol))

    h, w = imgs.shape[1], imgs.shape[2]
    grid = np.zeros((nlin * h, ncol * w, 3), dtype=np.uint8)

    for i in range(n):
        r = i // ncol
        c = i % ncol
        grid[r*h:(r+1)*h, c*w:(c+1)*w, :] = imgs[i]

    Image.fromarray(grid).save(out_path, format="JPEG", quality=95, optimize=True)
    print("[SAMPLE] saved:", out_path)


# =============================================================================
# 6) AUGMENTATION: cria OUTPUT_DIR/GAN_AUGMENTED_DATA com originais+geradas + manifest
# =============================================================================

def write_augmented_dataset(
    df: pd.DataFrame,
    generator: tf.keras.Model,
    id_to_class: Dict[int, int],
    cfg: CganConfig,
    out_root: Path,
) -> Path:
    """
    Cria subpasta GAN_AUGMENTED_DATA dentro de out_root:
      GAN_AUGMENTED_DATA/
        images/
        manifest.csv
        label_mapping.csv
        augmentation_config.json
        generator_path.txt

    Estratégia:
      - Copia todas as imagens originais (mantendo nome)
      - Gera (MULTIPLIER-1) imagens adicionais por imagem original, condicionadas ao MESMO label

    Assim o dataset final fica ~ MULTIPLIER * N.
    """
    aug_dir = out_root / "GAN_AUGMENTED_DATA"
    aug_images_dir = aug_dir / "images"
    ensure_dir(aug_images_dir)

    # Salva mapping (identity_id -> class_index)
    mapping_df = pd.DataFrame(
        [{"identity_id": k, "class_index": v} for k, v in sorted(id_to_class.items(), key=lambda x: x[1])]
    )
    mapping_path = aug_dir / "label_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)

    # Dump da config usada
    cfg_path = aug_dir / "augmentation_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    # 1) Copia originais e cria linhas de manifest
    rows: List[dict] = []
    n_original = len(df)

    print(f"[AUG] Copiando {n_original} imagens originais para {aug_images_dir} ...")
    copy_t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        src_path = Path(row.path)
        dst_path = aug_images_dir / Path(row.image_name).name

        # copy2 preserva timestamps quando possível
        shutil.copy2(src_path, dst_path)

        rows.append({
            "image_name": dst_path.name,
            "label": int(row.label),              # identidade ORIGINAL preservada
            "class_index": int(row.class_index),  # índice contíguo usado no GAN
            "source": "real",
            "parent_image": "",
            "created_utc": utc_now_iso(),
            "gan_model": "",
            "noise_seed": "",
        })

        if i % 5000 == 0:
            print(f"[AUG][COPY] {i}/{n_original} copiados...")

    print(f"[AUG][COPY] done em {time.time()-copy_t0:.1f}s")

    # 2) Gera sintéticas: (MULTIPLIER-1) por imagem
    assert cfg.MULTIPLIER >= 1, "MULTIPLIER deve ser >= 1."
    n_to_generate = (cfg.MULTIPLIER - 1) * n_original

    if n_to_generate <= 0:
        manifest_path = aug_dir / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        print("[AUG] MULTIPLIER=1 => nenhuma imagem sintética gerada.")
        print("[AUG] Manifest:", manifest_path)
        return aug_dir

    print(f"[AUG] Gerando {n_to_generate} imagens sintéticas (MULTIPLIER={cfg.MULTIPLIER}x) ...")

    gen_t0 = time.time()

    # Para manter a distribuição idêntica por identidade,
    # geramos (MULTIPLIER-1) por imagem original, com o mesmo label.
    # Isso é simples e garante total exatamente.
    gen_counter = 0

    # Geração em batches para eficiência
    batch_gen = max(16, cfg.BATCH_SIZE)  # reusa batch size (ou mínimo 16)
    quality = int(cfg.GEN_JPEG_QUALITY)

    # Percorre originais e, para cada um, gera K imagens condicionadas ao mesmo label
    k_per_image = cfg.MULTIPLIER - 1

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        identity_id = int(row.label)
        class_index = int(row.class_index)

        # geramos k_per_image imagens para esta identidade
        remaining = k_per_image
        while remaining > 0:
            b = min(batch_gen, remaining)
            remaining -= b

            # Ruído z ~ N(0,I) (Goodfellow 2014; DCGAN practice)
            noise = tf.random.normal((b, cfg.LATENT_DIM))
            labels = tf.fill((b, 1), tf.constant(class_index, dtype=tf.int32))

            gen_imgs = generator([noise, labels], training=False)  # [-1,1]
            gen_imgs = (gen_imgs + 1.0) / 2.0                     # [0,1]
            gen_imgs = tf.clip_by_value(gen_imgs, 0.0, 1.0)
            gen_imgs_u8 = tf.cast(gen_imgs * 255.0, tf.uint8).numpy()

            for j in range(b):
                gen_counter += 1
                fname = f"{cfg.GEN_PREFIX}_id{identity_id}_img{idx:06d}_g{gen_counter:08d}.jpg"
                out_path = aug_images_dir / fname

                Image.fromarray(gen_imgs_u8[j]).save(out_path, format="JPEG", quality=quality, optimize=True)

                rows.append({
                    "image_name": fname,
                    "label": identity_id,          # identidade ORIGINAL preservada
                    "class_index": class_index,    # índice contíguo do GAN
                    "source": "generated",
                    "parent_image": Path(row.image_name).name,
                    "created_utc": utc_now_iso(),
                    "gan_model": "cgan_models/generator.keras",
                    "noise_seed": "",              # opcional: você pode guardar seed/idx aqui
                })

                if gen_counter % 1000 == 0:
                    elapsed = time.time() - gen_t0
                    print(f"[AUG][GEN] {gen_counter}/{n_to_generate} geradas ({elapsed/60:.1f} min)")

    assert gen_counter == n_to_generate, f"Contagem divergente: {gen_counter} != {n_to_generate}"

    # 3) Escreve manifest final
    manifest_path = aug_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    print(f"[AUG] DONE. Total imagens no dataset final: {len(rows)} (orig={n_original}, gen={n_to_generate})")
    print("[AUG] Pasta final:", aug_dir)
    print("[AUG] Manifest:", manifest_path)
    print("[AUG] Mapping:", mapping_path)
    print("[AUG] Config:", cfg_path)

    return aug_dir


# =============================================================================
# 7) MAIN
# =============================================================================

def main():
    # Seeds
    set_seeds(CFG.RANDOM_SEED)

    # GPU (DirectML) validation
    configure_gpu_or_fail(require_gpu=CFG.REQUIRE_GPU)

    # Mixed precision (opcional)
    maybe_enable_mixed_precision(CFG.ENABLE_MIXED_PRECISION)

    input_dir = Path(CFG.INPUT_DATA_DIR)
    out_root = Path(CFG.OUTPUT_ROOT_DIR) if CFG.OUTPUT_ROOT_DIR else input_dir

    # Carrega manifest da ingestão (image_name + label + paths)
    df = load_ingestion_manifest(input_dir)

    # Mapeia identidades originais -> classes contíguas para o GAN
    df, id_to_class, class_to_id = make_label_mapping(df, CFG.MAX_IDENTITIES_FOR_GAN)

    # Subamostra opcional para treino do GAN
    df = maybe_subsample_images(df, CFG.MAX_IMAGES_FOR_GAN, CFG.RANDOM_SEED)

    # Dataset TF
    ds = build_tf_dataset(df, CFG.IMG_SIZE, CFG.BATCH_SIZE, CFG.BUFFER_SIZE)

    num_classes = len(id_to_class)
    print("[GAN] Treinando com num_classes =", num_classes)

    # Pasta de outputs do GAN (dentro do out_root, para ficar colado no pipeline)
    gan_out_dir = out_root / "cgan_training_outputs"
    ensure_dir(gan_out_dir)

    # Treina
    generator, discriminator = train_cgan(ds, num_classes, gan_out_dir, CFG)

    # Escreve dataset aumentado (orig + sintéticas) em GAN_AUGMENTED_DATA
    write_augmented_dataset(df, generator, id_to_class, CFG, out_root)

    print("\n[OK] Para treinar seu classificador consumindo o dataset aumentado, "
          "basta apontar o caminho de entrada para:")
    print("     ", out_root / "GAN_AUGMENTED_DATA")
    print("     (mesma ideia: images/ + manifest.csv)\n")


if __name__ == "__main__":
    main()


# =============================================================================
# 8) REFERÊNCIAS (para relatório)
# =============================================================================
"""
[1] Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014).
    "Generative Adversarial Nets." NeurIPS.

[2] Mirza, M., & Osindero, S. (2014).
    "Conditional Generative Adversarial Nets." arXiv:1411.1784.

[3] Radford, A., Metz, L., & Chintala, S. (2016).
    "Unsupervised Representation Learning with Deep Convolutional GANs (DCGAN)." arXiv:1511.06434.

[4] Kingma, D. P., & Ba, J. (2015).
    "Adam: A Method for Stochastic Optimization." ICLR.

[5] Ioffe, S., & Szegedy, C. (2015).
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML.

[6] Nair, V., & Hinton, G. E. (2010).
    "Rectified Linear Units Improve Restricted Boltzmann Machines." ICML.

[7] Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013).
    "Rectifier Nonlinearities Improve Neural Network Acoustic Models." ICML Workshop.

[8] Micikevicius, P., Narang, S., Alben, J., et al. (2018).
    "Mixed Precision Training." ICLR.
"""
