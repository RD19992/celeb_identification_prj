## -*- coding: utf-8 -*-
""""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Código para ingestão do dataset CELEB-A e preparação para processamento
"""

# -*- coding: utf-8 -*-
"""
CelebA - Ingestão RGB para CNN (resolução arbitrária)
- Usa dataset já baixado em: data/celeba/img_align_celeba
- Usa labels já existentes (identity_CelebA.txt e/ou labels.csv)
- Gera novo conjunto de imagens RGB redimensionadas e salva em:
  data/cnn_identification_authorization/...

Requisitos: opencv-python, pandas, matplotlib
"""



import os
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIGURAÇÕES
# =========================
TARGET_SIZE = 128  # resolução arbitrária (comece com 256)
SQUARE_CROP = True  # evita distorção (crop central p/ quadrado antes do resize)
JPEG_QUALITY = 95

# Subamostragem opcional (para testes rápidos)
PERCENT_IMAGES = 1.00   # 1.0 = 100%
MAX_IMAGES = None       # ex.: 5000 (ou None)

# (Opcional) filtrar para as classes mais frequentes (para reduzir custo)
TOP_CLASS_FRACTION = 0.20  # 1.0 = mantém todas as classes; ex.: 0.20 mantém top 20%

# Amostras visuais
N_SHOW_SAMPLES = 5
RANDOM_SEED = 42

# Se False, não reprocessa imagens já existentes no output
OVERWRITE = False


# =========================
# PATHS
# =========================
HERE = Path(__file__).resolve().parent
# Script está em: <project_root>/data/cnn_identification_authorization/
PROJECT_ROOT = HERE.parents[1]  # sobe para <project_root>

CELEBA_DIR = PROJECT_ROOT / "celeba"
RAW_IMAGES_DIR = CELEBA_DIR / "img_align_celeba"
IDENTITY_FILE = CELEBA_DIR / "identity_CelebA.txt"
LABELS_CSV = CELEBA_DIR / "labels.csv"

OUTPUT_DIR = HERE / f"celeba_rgb_{TARGET_SIZE}x{TARGET_SIZE}"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.csv"


# =========================
# UTIL
# =========================
def _fail(msg: str) -> None:
    raise FileNotFoundError(msg)


def validate_inputs() -> None:
    if not CELEBA_DIR.exists():
        _fail(f"Pasta não encontrada: {CELEBA_DIR}")
    if not RAW_IMAGES_DIR.exists():
        _fail(f"Pasta de imagens não encontrada: {RAW_IMAGES_DIR}")
    if not IDENTITY_FILE.exists() and not LABELS_CSV.exists():
        _fail(
            "Não encontrei identity_CelebA.txt nem labels.csv em data/celeba. "
            "Garanta que um deles exista."
        )

    n_imgs = len(list(RAW_IMAGES_DIR.glob("*.jpg")))
    if n_imgs == 0:
        _fail(f"Nenhuma imagem .jpg encontrada em: {RAW_IMAGES_DIR}")

    print("[OK] Inputs encontrados")
    print(" - CELEBA_DIR:", CELEBA_DIR)
    print(" - RAW_IMAGES_DIR:", RAW_IMAGES_DIR)
    print(" - #imagens jpg:", n_imgs)
    print(" - identity_CelebA.txt existe?", IDENTITY_FILE.exists())
    print(" - labels.csv existe?", LABELS_CSV.exists())


def load_labels() -> pd.DataFrame:
    """
    Retorna DataFrame com colunas: image_name, label (int).
    Preferência: labels.csv (se existir). Caso contrário, parse do identity_CelebA.txt.
    """
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        # tolera variações pequenas
        if "image_name" not in df.columns or "label" not in df.columns:
            _fail(f"{LABELS_CSV} existe, mas não tem colunas image_name/label.")
        df["label"] = df["label"].astype(int)
        return df[["image_name", "label"]].copy()

    # fallback: parse do identity file
    rows = []
    with IDENTITY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            fname, ident = line.strip().split()
            rows.append((fname, int(ident)))
    df = pd.DataFrame(rows, columns=["image_name", "label"])
    return df


def filter_top_classes(df: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    if not (0 < top_fraction <= 1.0):
        raise ValueError("TOP_CLASS_FRACTION deve estar em (0, 1].")

    if top_fraction >= 1.0:
        return df

    counts = df["label"].value_counts()
    n_classes = len(counts)
    k = max(1, int(math.ceil(top_fraction * n_classes)))
    top_labels = set(counts.nlargest(k).index.tolist())

    df2 = df[df["label"].isin(top_labels)].reset_index(drop=True)
    print(f"[INFO] Mantendo top classes: {k}/{n_classes} ({100*k/n_classes:.1f}%)")
    print(f"[INFO] Imagens após filtro de classe: {len(df2)}/{len(df)} ({100*len(df2)/len(df):.1f}%)")
    return df2


def sample_df(df: pd.DataFrame,
              percent: float,
              max_images: Optional[int],
              seed: int) -> pd.DataFrame:
    if not (0 < percent <= 1.0):
        raise ValueError("PERCENT_IMAGES deve estar em (0, 1].")

    n_total = len(df)
    n_use = int(math.ceil(n_total * percent))
    if max_images is not None:
        n_use = min(n_use, int(max_images))

    df2 = df.sample(n=n_use, random_state=seed).reset_index(drop=True)
    print(f"[INFO] Subamostra: {len(df2)}/{n_total} imagens")
    return df2


def center_square_crop_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_bgr[y0:y0+side, x0:x0+side]


def read_resize_save_rgb(src_path: Path,
                         dst_path: Path,
                         target_size: int,
                         square_crop: bool,
                         overwrite: bool,
                         jpeg_quality: int) -> Tuple[bool, str]:
    """
    Lê BGR via OpenCV, (opcional) crop central para quadrado, resize p/ target_size, salva JPEG.
    Retorna (ok, error_msg).
    """
    try:
        if dst_path.exists() and not overwrite:
            return True, ""

        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            return False, "cv2.imread retornou None"

        if square_crop:
            img = center_square_crop_bgr(img)

        # resize
        interp = cv2.INTER_AREA if (img.shape[0] > target_size or img.shape[1] > target_size) else cv2.INTER_CUBIC
        img = cv2.resize(img, (target_size, target_size), interpolation=interp)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(dst_path),
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ok:
            return False, "cv2.imwrite falhou"
        return True, ""
    except Exception as e:
        return False, repr(e)


def show_random_images(df: pd.DataFrame,
                       images_dir: Path,
                       title_prefix: str,
                       n: int,
                       seed: int) -> None:
    rng = random.Random(seed)
    sample_rows = df.sample(n=min(n, len(df)), random_state=seed)

    plt.figure(figsize=(3*n, 3))
    for i, row in enumerate(sample_rows.itertuples(index=False), start=1):
        img_path = images_dir / row.image_name
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plt.subplot(1, n, i)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"{title_prefix}\nID: {row.label}")
    plt.tight_layout()
    plt.show()


def build_resized_dataset(df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    n_ok = 0
    n_fail = 0

    for row in df.itertuples(index=False):
        src = RAW_IMAGES_DIR / row.image_name
        if not src.exists():
            manifest_rows.append({
                "image_name": row.image_name,
                "label": int(row.label),
                "src": str(src),
                "dst": "",
                "ok": False,
                "error": "arquivo fonte não existe"
            })
            n_fail += 1
            continue

        dst = OUTPUT_IMAGES_DIR / row.image_name  # mantém mesmo nome
        ok, err = read_resize_save_rgb(
            src_path=src,
            dst_path=dst,
            target_size=TARGET_SIZE,
            square_crop=SQUARE_CROP,
            overwrite=OVERWRITE,
            jpeg_quality=JPEG_QUALITY
        )

        manifest_rows.append({
            "image_name": row.image_name,
            "label": int(row.label),
            "src": str(src),
            "dst": str(dst),
            "ok": bool(ok),
            "error": err
        })

        if ok:
            n_ok += 1
        else:
            n_fail += 1

        if (n_ok + n_fail) % 5000 == 0:
            print(f"[PROGRESS] processadas: {n_ok+n_fail} | ok={n_ok} | fail={n_fail}")

    manifest = pd.DataFrame(manifest_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_MANIFEST, index=False)
    print("[DONE] Output em:", OUTPUT_DIR)
    print("[DONE] Imagens em:", OUTPUT_IMAGES_DIR)
    print("[DONE] Manifest:", OUTPUT_MANIFEST)
    print(f"[DONE] ok={n_ok} | fail={n_fail}")
    return manifest


def show_random_generated_samples(manifest: pd.DataFrame, n: int, seed: int) -> None:
    ok_df = manifest[manifest["ok"] == True].copy()
    if ok_df.empty:
        print("[WARN] Nenhuma imagem gerada com sucesso para amostrar.")
        return

    sample_rows = ok_df.sample(n=min(n, len(ok_df)), random_state=seed)

    plt.figure(figsize=(3*n, 3))
    for i, row in enumerate(sample_rows.itertuples(index=False), start=1):
        img_bgr = cv2.imread(str(row.dst), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        plt.subplot(1, n, i)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"GERADA\nID: {row.label}")
    plt.tight_layout()
    plt.show()

    # também imprime o par (arquivo, id) para conferência rápida
    print("[SAMPLES - generated]")
    print(sample_rows[["image_name", "label", "dst"]].to_string(index=False))


def main():
    random.seed(RANDOM_SEED)

    validate_inputs()

    df = load_labels()
    print("[INFO] Labels carregados:", df.shape)
    print(df.head())

    df = filter_top_classes(df, TOP_CLASS_FRACTION)
    df = sample_df(df, PERCENT_IMAGES, MAX_IMAGES, RANDOM_SEED)

    # Amostras do RAW (como no seu script atual)
    print("[INFO] Amostras RAW")
    show_random_images(df, RAW_IMAGES_DIR, title_prefix="RAW", n=N_SHOW_SAMPLES, seed=RANDOM_SEED)

    # Gera dataset redimensionado RGB e salva
    manifest = build_resized_dataset(df)

    # Amostras do GERADO + IDs
    print("[INFO] Amostras GERADAS")
    show_random_generated_samples(manifest, n=N_SHOW_SAMPLES, seed=RANDOM_SEED)


if __name__ == "__main__":
    main()
