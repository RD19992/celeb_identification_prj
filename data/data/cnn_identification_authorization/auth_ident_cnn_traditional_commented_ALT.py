# -*- coding: utf-8 -*-
""""
EACH_USP: SIN-5016 - Aprendizado de Máquina
Laura Silva Pelicer
Renan Rios Diniz

Código de avaliação de modelo com tarefas de identificação e autenticação
Consome parâmetros de modelo treinado
CNN Tradicional
"""

# ==================================================================================================
# NOTAS  / REFERÊNCIAS
# --------------------------------------------------------------------------------------------------
# Este script implementa um pipeline de *avaliação* (não-treino) para duas tarefas comuns em reconhecimento facial:
#   (1) Identificação (classificação multi-classe: 'quem é?')
#   (2) Autenticação / verificação (binária: 'é a mesma pessoa?') via pares e um limiar (threshold).
#
# A lógica central segue a família de abordagens 'embedding + similaridade' popularizada em verificação facial
# (por exemplo, FaceNet / ArcFace), mas aqui reaproveita a saída do modelo (CNN Keras) para:
#   - gerar probabilidades por classe P(x) (softmax);
#   - extrair um embedding (ativação interna) e L2-normalizar;
#   - comparar pares por: (i) cosseno (dot em embeddings normalizados), (ii) dot de probas,
#     (iii) regra discreta de 'mesmo ID predito' + confiança.
#
# A tarefa de autenticação por pares ('same/different') é o protocolo clássico de benchmarks como LFW.
# A lógica moderna costuma operar com *embeddings* L2-normalizados e similaridade cosseno (ou distância Euclidiana)
# e escolhe um limiar para equilibrar taxas de falso positivo/negativo (Fawcett, 2006; Schroff et al., 2015).
# A amostragem de pares aqui evita o custo quadrático de enumerar todos os pares possíveis (O(N^2)) em datasets grandes.
#
# Sobre os hiperparâmetros 'de avaliação' (não do treino):
#   • sampling de pares (pos_pairs_per_class, neg_pairs_total, sample_pairs_if_large) controla custo O(N^2).
#   • threshold_grid_q define a granularidade de busca do limiar por quantis (trade-off custo vs resolução).
#   • tune_metric escolhe o critério (F1 / balanced accuracy / accuracy), afetando o ponto de operação.
#   • macro_auc (por identidade) corrige o viés do micro-AUC quando há classes desbalanceadas (#imagens por pessoa).
#

#Referências
# C. M. Bishop, Pattern Recognition and Machine Learning. Springer, 2006.
# K. P. Murphy, Machine Learning: A Probabilistic Perspective. MIT Press, 2012.
# T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006.
# C. J. van Rijsbergen, Information Retrieval, 2nd ed. Butterworth-Heinemann, 1979. (F-measure)
# D. M. W. Powers, “Evaluation: From Precision, Recall and F-Measure to ROC…,” Journal of Machine Learning Technologies, 2011.
# B. Schölkopf and A. J. Smola, Learning with Kernels. MIT Press, 2002.
# G. B. Huang et al., “Labeled Faces in the Wild…,” UMass Amherst, Tech. Rep. 07-49, 2007.
# F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet…,” in Proc. IEEE CVPR, 2015.
# J. Deng et al., “ArcFace…,” in Proc. IEEE CVPR, 2019.
# H. He and E. A. Garcia, “Learning from Imbalanced Data,” IEEE TKDE, vol. 21, no. 9, pp. 1263–1284, 2009.
# M. Sokolova and G. Lapalme, “A systematic analysis of performance measures…,” Information Processing & Management, 2009.
# P. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” in Proc. IJCAI, 1995.
# S. Kaufman, S. Rosset, and C. Perlich, “Leakage in Data Mining: Formulation, Detection, and Avoidance,” in Proc. ACM KDD, 2012.
# E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” Nature, vol. 323, pp. 533–536, 1986.
# Wang, Feng, et al. "Normface: L2 hypersphere embedding for face verification." Proceedings of the 25th ACM international conference on Multimedia. 2017.
# Krizhevsky, Alex, and Geoff Hinton. "Convolutional deep belief networks on cifar-10." Unpublished manuscript 40.7 (2010): 1-9.
# K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification,” in Proc. ICCV, 2015.
# J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv:1607.06450, 2016.
#
#
# Nota prática: quando este script roda no modo CNN Keras, a leitura/normalização das imagens ocorre via tf.data
# e decode_jpeg/resize, com padrão comum de normalização por canal (mean/std) usado em pipelines de visão.



# ==================================================================================================

from __future__ import annotations

import sys
import math
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG DO SCRIPT (ajuste aqui)
#
# (Referências) A ideia de 'ponto de operação' escolhido por limiar vem da análise ROC/PR (Fawcett, 2006) e
# do fato de que em verificação a métrica de interesse costuma ser sensível ao custo relativo FP/FN.
# Para classes desbalanceadas, balanced accuracy / macro averaging são recomendações recorrentes
# (Sokolova & Lapalme, 2009; He & Garcia, 2009).
#
# Observação: estes hiperparâmetros afetam APENAS a avaliação/decisão; não alteram pesos do modelo.
#
# ============================================================

import os
import json
import pandas as pd
import tensorflow as tf
from PIL import Image

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SCRIPT_CONFIG = {
    # nome do arquivo salvo pelo script de treino (dentro do run_dir)
    "model_payload_file": "models/final_model_best.keras",

    # opcional: apontar diretamente para um run_dir específico (onde estão hyperparameters.json, label2idx.json, models/)
    # se None, o script tentará auto-detectar o run mais recente em ./celeba_rgb_128x128/runs/ e vizinhança.
    "run_dir_override": None,

    # se quiser sobrescrever o caminho do joblib HOG/dataset
    "dataset_path_override": None,  # ex.: r"C:\\...\\celeba_hog_128x128_o9.joblib"

    # aleatoriedade local deste script (não altera o split do treino, só prints/amostragens)
    "seed": 123,

    # identificação
    "one_vs_all_n_classes": 10,

    "roc": {
        "enable": True,
        "save_png": True,
        "show_plots": True,
        "png_name": "roc_ova_10classes.png",
    },

    # OBS: "macro_auc" abaixo mede AUC de IDENTIFICAÇÃO (one-vs-rest por classe).
    # Isso NÃO é AUC de autenticação (same vs different).
    "macro_auc": {
        "enable": False,
        "use_fast_auc": True,  # True: usa roc_auc_fast (mais rápido); False: usa roc_curve_simple
        "progress_every": 50,    # imprime status a cada N classes
    },

    # autenticação
    "auth": {
        "tune_metric": "f1",  # "f1" | "balanced_acc" | "acc"

        # tuning do threshold (amostragem):
        "pos_pairs_per_class": 50,  # tenta gerar até isso por classe (se houver amostras)
        "neg_pairs_total": 20000,  # negativos totais para tuning
        "threshold_grid_q": 401,  # quantis para varrer threshold (>=101 recomendado)

        # avaliação em pares:
        # modo "auto": se N for pequeno faz "full"; caso contrário faz "sample"
        "eval_mode": "auto",  # "auto" | "full" | "sample"
        "full_if_n_leq": 2500,  # se N <= isso, tenta full (O(N^2))
        "sample_pairs_if_large": 300000,  # #pares amostrados se N grande
        "pairs_fraction": 1.0,  # 0..1: fração dos pares amostrados usada em avaliação/AUC (1.0 = 100%)

        # matrizes por âncora
        "n_anchor_images": 10,

        # =====================================================
        # Macro AUC (AUTENTICAÇÃO) por identidade
        # =====================================================
        # O AUC global acima (same vs different) é um "micro-AUC":
        # cada PAR conta igualmente, então classes com mais imagens pesam mais.
        # Este bloco calcula um "macro-AUC":
        #   1) para cada identidade/classe c, amostra pares POS (c vs c) e NEG (c vs ~c)
        #   2) calcula AUC(c)
        #   3) tira a média simples dos AUC(c) em todas as classes
        "macro_auc": {
            "enable": True,
            "pos_pairs_per_class": 30,
            "neg_pairs_per_class": 60,
            "classes_fraction": 1.0,  # 1.0 = usa todas as classes; <1.0 amostra subconjunto
            "max_classes": 0,         # 0 = sem limite; >0 limita #classes (amostradas) por custo
            "progress_every": 100,
        },
    },

    # modo interativo (input)
    "enable_interactive_queries": False,
}


def _resolve_out_dir() -> Path:
    """Resolve o diretório de execução (out_dir) onde estão:
      - hyperparameters.json
      - label2idx.json
      - models/*.keras

    Para manter a lógica do script original (tudo 'na mesma pasta'),
    aqui 'a mesma pasta' passa a ser o run_dir gerado pelo treino da CNN tradicional.
    """
    base_dir = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd())

    override = SCRIPT_CONFIG.get("run_dir_override") or os.environ.get("AUTH_IDENT_RUN_DIR") or os.environ.get("CNN_RUN_DIR")
    if override:
        p = Path(str(override)).expanduser()
        return p

    # Candidatos comuns (seguindo cnn_traditional.py)
    candidates = [
        base_dir,
        base_dir / "runs",
        base_dir / "celeba_rgb_128x128" / "runs",
    ]

    # Também tenta descobrir qualquer 'runs' até 2 níveis abaixo
    try:
        for p in base_dir.glob("*/runs"):
            candidates.append(p)
        for p in base_dir.glob("*/*/runs"):
            candidates.append(p)
    except Exception:
        pass

    run_dirs: List[Path] = []
    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue

        # Caso root já seja um run_dir
        if (root / "hyperparameters.json").exists() and (root / "label2idx.json").exists():
            run_dirs.append(root)
            continue

        # Caso root seja o diretório 'runs'
        try:
            for sub in root.iterdir():
                if sub.is_dir() and (sub / "hyperparameters.json").exists() and (sub / "label2idx.json").exists():
                    run_dirs.append(sub)
        except Exception:
            pass

    if not run_dirs:
        return base_dir

    def score(rd: Path):
        model_dir = rd / "models"
        has_model = False
        try:
            has_model = model_dir.exists() and any(model_dir.glob("*.keras"))
        except Exception:
            has_model = False
        try:
            mtime = max((rd / "hyperparameters.json").stat().st_mtime, rd.stat().st_mtime)
        except Exception:
            mtime = rd.stat().st_mtime if rd.exists() else 0.0
        return (1 if has_model else 0, mtime)

    run_dirs.sort(key=score, reverse=True)
    return run_dirs[0]


def _resolve_model_path(out_dir: Path, cfg_train: Dict[str, Any]) -> Path:
    """Resolve caminho do modelo Keras a partir de:
      - SCRIPT_CONFIG['model_payload_file']
      - cfg_train['FINAL_MODEL_FILENAME'] (se existir)
      - fallback: *.keras mais recente em out_dir/models/
    """
    rel = str(SCRIPT_CONFIG.get("model_payload_file", "models/final_model_best.keras"))
    final_name = cfg_train.get("FINAL_MODEL_FILENAME")
    if final_name and rel.startswith("models/"):
        rel = f"models/{final_name}"

    model_path = out_dir / rel
    if model_path.exists():
        return model_path

    models_dir = out_dir / "models"
    if models_dir.exists():
        cands = sorted(models_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            print(f"[WARN] Modelo padrão não encontrado ({model_path.name}); usando o mais recente em models/: {cands[0].name}")
            return cands[0]

    raise FileNotFoundError(f"Modelo Keras não encontrado: {model_path}")


# ============================================================
# IO DO DATASET (legado HOG joblib)
# ============================================================

def carregar_dataset_joblib(path: str, *, label2idx: Optional[Dict[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Carrega dataset.

    Suporta 2 formatos:
      1) .joblib (original, para MLP/HOG)  -> mantém comportamento;
      2) manifest.csv (CNN tradicional)    -> retorna caminhos (string) e y reindexado.

    Observação: no caso do manifest.csv, a remapeação label->y depende de label2idx.json
    (passado via label2idx).
    """
    p = Path(path)
    if p.is_dir():
        cand = p / "manifest.csv"
        if cand.exists():
            p = cand

    if p.suffix.lower() == ".csv":
        return carregar_dataset_manifest_csv(str(p), label2idx=label2idx)

    # joblib (fluxo original)
    import joblib
    data = joblib.load(str(p))
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return X, y, meta


def carregar_dataset_manifest_csv(manifest_csv: str, *, label2idx: Optional[Dict[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Lê o manifest.csv para o caso CNN tradicional.

    Retorna:
      - X: array (N,) de caminhos para imagem (string)
      - y: array (N,) com rótulos reindexados (0..C-1) compatíveis com o modelo
      - meta: dicionário com chaves 'paths' (iguais a X) e 'ids' (rótulos originais)

    Espera colunas no CSV (conforme o script de treino da CNN):
      - dst: caminho do arquivo de imagem (absoluto ou relativo ao diretório do manifest)
      - label: id da classe/identidade (numérico)
      - ok (opcional): 1/0 ou True/False para filtrar exemplos válidos

    A remapeação label->y é feita via label2idx.json gerado no treino (passado em label2idx).
    Se label2idx não for fornecido, tenta carregar 'label2idx.json' no mesmo diretório do manifest
    (fallback útil quando o script é executado dentro do run_dir).
    """
    import pandas as pd

    p = Path(manifest_csv).resolve()
    if not p.exists():
        raise FileNotFoundError(f"manifest.csv não encontrado: {p}")

    df = pd.read_csv(p)

    # ok (opcional)
    if "ok" in df.columns:
        okv = df["ok"]
        if okv.dtype == bool:
            df = df[okv]
        else:
            df = df[okv.astype(int) == 1]

    if "dst" not in df.columns or "label" not in df.columns:
        raise ValueError("manifest.csv deve ter colunas 'dst' e 'label'.")

    # resolver caminhos (se relativos ao diretório do manifest)
    base_dir = p.parent
    dst = df["dst"].astype(str).tolist()
    dst_resolved: List[str] = []
    for s in dst:
        sp = Path(s)
        if not sp.is_absolute():
            sp = (base_dir / sp).resolve()
        dst_resolved.append(str(sp))

    # label2idx
    if label2idx is None:
        cand = base_dir / "label2idx.json"
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # json pode ter chaves str -> converter
            label2idx = {int(k): int(v) for k, v in raw.items()}
        else:
            label2idx = {}

    # mapear labels originais para y reindexado
    labels_orig = df["label"].astype(int).to_numpy()
    y_list: List[int] = []
    x_list: List[str] = []
    id_list: List[int] = []

    for spath, lab in zip(dst_resolved, labels_orig):
        if label2idx:
            if int(lab) not in label2idx:
                continue
            yv = int(label2idx[int(lab)])
        else:
            # fallback: sem label2idx, usa label como está (pode não bater com o modelo!)
            yv = int(lab)

        if not Path(spath).exists():
            continue

        x_list.append(spath)
        y_list.append(yv)
        id_list.append(int(lab))

    X = np.asarray(x_list, dtype=str)
    y = np.asarray(y_list, dtype=np.int64)

    meta: Dict[str, Any] = {
        "paths": X.copy(),
        "ids": np.asarray(id_list, dtype=np.int64),
    }
    return X, y, meta


def selecionar_classes_elegiveis(y: np.ndarray, min_amostras: int) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    return classes[counts >= int(min_amostras)].astype(np.int64, copy=False)


def amostrar_classes(classes: np.ndarray, frac: float, seed: int) -> np.ndarray:
    frac = float(frac)
    classes = np.asarray(classes, dtype=np.int64)
    if frac >= 0.999999:
        return np.array(classes, copy=True)
    rng = np.random.default_rng(int(seed))
    n = len(classes)
    k = max(1, int(np.ceil(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    return np.sort(classes[idx]).astype(np.int64, copy=False)


def filtrar_por_classes(X: np.ndarray, y: np.ndarray, classes_permitidas: np.ndarray,
                        paths: Optional[np.ndarray] = None,
                        ids: Optional[np.ndarray] = None,
                        idx_global: Optional[np.ndarray] = None):
    mask = np.isin(y, classes_permitidas)
    Xf = X[mask]
    yf = y[mask]
    pf = paths[mask] if paths is not None else None
    idf = ids[mask] if ids is not None else None
    igf = idx_global[mask] if idx_global is not None else None
    return Xf, yf, pf, idf, igf


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Aplica padronização.

    - Se X for um vetor de *caminhos de arquivo* (dtype string/object), não padroniza aqui
      (a normalização é aplicada on-the-fly no pipeline TensorFlow em inferência).
    - Para vetores (MLP/HOG): comportamento idêntico ao original.
    - Para imagens (CNN) em N,H,W,C:
        * se X for uint8/uint16, assume [0,255] e converte para [0,1];
        * aplica (X-mean)/std com broadcasting em C;
        * é idempotente: se parecer já padronizado, não padroniza de novo.
    """
    X = np.asarray(X)

    # Caso CNN (lazy): X é array de caminhos -> padronização ocorre na leitura via tf.data
    if X.ndim == 1 and X.dtype.kind in ("U", "S", "O"):
        return X

    Xf = X.astype(np.float32, copy=False)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, -1)) if Xf.ndim == 4 else np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, -1)) if Xf.ndim == 4 else np.asarray(std, dtype=np.float32)

    # Heurística simples de idempotência p/ imagens:
    if Xf.ndim == 4:
        # se já parece em escala "padronizada" (valores típicos perto de 0)
        mx = float(np.nanmax(np.abs(Xf))) if Xf.size else 0.0
        if mx < 6.0:
            return Xf

        # se parece em [0,255], converte p/ [0,1]
        if float(np.nanmax(Xf)) > 3.0:
            Xf = Xf / np.float32(255.0)

        return (Xf - mean_arr) / std_arr

    # Vetores (MLP/HOG)
    return (Xf - mean_arr) / std_arr


def stable_softmax(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    Zm = Z - Z.max(axis=1, keepdims=True)
    np.exp(Zm, out=Zm)
    Zm /= Zm.sum(axis=1, keepdims=True)
    return Zm


def _row_norm_forward(A: np.ndarray, eps: float):
    A = A.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(A * A, axis=1, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return A * inv, inv.astype(np.float32, copy=False)


def _col_norm_forward(W: np.ndarray, eps: float):
    W = W.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(W * W, axis=0, keepdims=True)) + float(eps)
    inv = 1.0 / norms
    return W * inv, inv.astype(np.float32, copy=False)


def output_logits_forward(
        A1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        act_output: str,
        scale: float,
        eps: float,
        use_bias: bool,
):
    act_output = str(act_output).lower().strip()

    if act_output in ("softmax", "linear", "linear_softmax"):
        Z2 = A1 @ W2 + b2
        return Z2

    if act_output == "cosine_softmax":
        A_hat, _ = _row_norm_forward(A1, eps=eps)
        W_hat, _ = _col_norm_forward(W2, eps=eps)
        Z2 = float(scale) * (A_hat @ W_hat)
        if bool(use_bias):
            Z2 = Z2 + b2
        return Z2

    raise ValueError(f"act_output desconhecida: {act_output}")


def tanh_custom(Z: np.ndarray):
    return np.tanh(Z.astype(np.float32, copy=False)).astype(np.float32, copy=False)


# (ReLU) Função de ativação retificadora (Nair & Hinton, 2010; He et al., 2015).
# ReLU tende a reduzir saturação (vs. tanh/sigmoid) e melhora o fluxo de gradiente em redes profundas.
def relu_custom(Z: np.ndarray):
    Z = Z.astype(np.float32, copy=False)
    return np.maximum(np.float32(0.0), Z).astype(np.float32, copy=False)


def activation_forward(Z: np.ndarray, act: str):
    act = str(act).lower().strip()
    if act == "tanh":
        return tanh_custom(Z)
    if act == "relu":
        return relu_custom(Z)
    raise ValueError(f"Ativação desconhecida: {act}")


# (LayerNorm) Implementação de normalização por amostra (Ba et al., 2016).
# Útil para estabilizar distribuições internas e melhorar condicionamento de otimização, especialmente quando
# batch statistics não são desejáveis (comparar com BatchNorm de Ioffe & Szegedy, 2015 — não usado aqui).
def layernorm_forward(Z: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float):
    Z = Z.astype(np.float32, copy=False)
    mu = Z.mean(axis=1, keepdims=True)
    var = Z.var(axis=1, keepdims=True)
    invstd = 1.0 / np.sqrt(var + np.float32(eps))
    xhat = (Z - mu) * invstd
    out = xhat * gamma + beta
    return out.astype(np.float32, copy=False)


def mlp_forward_inference(
    X: np.ndarray,
    modelo: Any,
    inference_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass de inferência compatível com 2 cenários:

    1) modelo == dict (payload do MLP/HOG original):
       - Mantém exatamente o comportamento esperado no auth_ident original.

    2) modelo == tf.keras.Model (CNN tradicional):
       - X pode ser:
         a) array (N,) de caminhos de arquivo (string/object) -> lê/normaliza via tf.data;
         b) array numérico (N,H,W,C) já carregado -> faz predict direto.
       - Retorna:
         P: (N, C) probabilidades (softmax)
         A1_pre: (N, D) embedding (saída da camada indicada por inference_params['embedding_layer_name'])
    """
    X = np.asarray(X)

    # ===== Caso original (MLP/HOG) =====
    if isinstance(modelo, dict) and "weights" in modelo:
        # Mantém o comportamento do script original
        P, A1_pre = mlp_forward(
            X,
            modelo,
            training=False,
            inference_only=True,
            inference_params=inference_params,
        )
        return P.astype(np.float32), A1_pre.astype(np.float32)

    # ===== Caso CNN (Keras) =====
    if hasattr(modelo, "predict") and hasattr(modelo, "layers"):
        batch_size = int(inference_params.get("batch_size", 64))
        embedding_layer_name = str(inference_params.get("embedding_layer_name", "dense"))

        # model que devolve [probs, embedding] em 1 só forward
        combined = inference_params.get("_combined_model", None)
        if combined is None:
            try:
                emb_out = modelo.get_layer(embedding_layer_name).output
            except Exception:
                # fallback: penúltima camada
                emb_out = modelo.layers[-2].output
            combined = tf.keras.Model(inputs=modelo.input, outputs=[modelo.output, emb_out])
            inference_params["_combined_model"] = combined

        # X como caminhos (lazy)
        if X.ndim == 1 and X.dtype.kind in ("U", "S", "O"):
            img_size = int(inference_params.get("img_size", 128))
            channels = int(inference_params.get("channels", 3))
            assume_ingested_size = bool(inference_params.get("assume_ingested_size", True))
            jpeg_dct_method = str(inference_params.get("jpeg_dct_method", "INTEGER_FAST"))

            mean_vec = np.asarray(inference_params.get("norm_mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape((-1,))
            std_vec  = np.asarray(inference_params.get("norm_std",  [1.0, 1.0, 1.0]), dtype=np.float32).reshape((-1,))
            # broadcast correto: (H,W,C) +/- (C,) -> (H,W,C) (sem criar dimensão extra 1)
            mean_tf = tf.constant(mean_vec, dtype=tf.float32)
            std_tf  = tf.constant(std_vec, dtype=tf.float32)

            def _parse(path_tensor: tf.Tensor) -> tf.Tensor:
                img_bytes = tf.io.read_file(path_tensor)
                img = tf.image.decode_jpeg(img_bytes, channels=channels, dct_method=jpeg_dct_method)
                img = tf.image.convert_image_dtype(img, tf.float32)
                if not assume_ingested_size:
                    img = tf.image.resize(img, [img_size, img_size], antialias=True)
                img = (img - mean_tf) / std_tf
                return img

            # tf.data / AUTOTUNE: pipeline de entrada eficiente (streaming + paralelismo) para inferência em lotes.
            # Este padrão (Dataset -> map(decode/resize) -> batch -> prefetch) é recomendado para throughput e
            # desacoplamento CPU (decoding) / GPU (inference). Ver também a descrição do TensorFlow como sistema
            # de grafos de fluxo de dados (Abadi et al., 2016).
            paths_tf = tf.convert_to_tensor(X.astype(str))
            ds = tf.data.Dataset.from_tensor_slices(paths_tf)
            ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            P, A1_pre = combined.predict(ds, verbose=0)
        else:
            # numérico já carregado (idealmente já normalizado por apply_standardizer)
            P, A1_pre = combined.predict(X, batch_size=batch_size, verbose=0)

        return np.asarray(P, dtype=np.float32), np.asarray(A1_pre, dtype=np.float32)

    raise TypeError("Formato de 'modelo' não suportado para inferência.")


def predict_labels(X: np.ndarray, classes: np.ndarray, modelo: Dict[str, Any], inference_params: Dict[str, Any]):
    P, _ = mlp_forward_inference(X, modelo, inference_params)
    idx = np.argmax(P, axis=1).astype(np.int64, copy=False)
    return classes[idx], P


def extract_embeddings(X: np.ndarray, modelo: Dict[str, Any], inference_params: Dict[str, Any]) -> np.ndarray:
    """
    Embedding = ativação da camada escondida (A1_pre) L2-normalizada (cosine-ready).
    """
    _, A1_pre = mlp_forward_inference(X, modelo, inference_params)
    A_hat, _ = _row_norm_forward(A1_pre, eps=float(inference_params.get("cosine_softmax_eps", 1e-8)))
    return A_hat.astype(np.float32, copy=False)


# ============================================================
# Métricas / Confusões
# ============================================================

# Referências para métricas
# M. Sokolova and G. Lapalme, “A systematic analysis of performance measures…,” Information Processing & Management, 2009.
# D. M. W. Powers, “Evaluation: From Precision, Recall and F-Measure to ROC” Journal of Machine Learning Technologies, 2011.
# T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_one_vs_all(y_true: np.ndarray, y_pred: np.ndarray, pos_class: int) -> Dict[str, int]:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    tpos = (y_true == int(pos_class))
    ppos = (y_pred == int(pos_class))

    TP = int(np.sum(tpos & ppos))
    FP = int(np.sum((~tpos) & ppos))
    FN = int(np.sum(tpos & (~ppos)))
    TN = int(np.sum((~tpos) & (~ppos)))
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


def print_confusion_binary(title: str, cm: Dict[str, int]):
    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    print(f"\n{title}")
    print("            Pred=0    Pred=1")
    print(f"True=0      {TN:6d}    {FP:6d}")
    print(f"True=1      {FN:6d}    {TP:6d}")
    denom = TP + TN + FP + FN
    acc = (TP + TN) / denom if denom > 0 else 0.0
    print(f"acc={acc:.4f} | TP={TP} FP={FP} FN={FN} TN={TN}")


# ============================================================
# Autenticação: threshold tuning + avaliação de pares
# ============================================================

#Referência para autenticação por pares
# G. B. Huang et al., “Labeled Faces in the Wild…,” UMass Amherst, Tech. Rep. 07-49, 2007.

def _sample_positive_pairs_per_class(y: np.ndarray, rng: np.random.Generator, per_class: int):
    y = np.asarray(y, dtype=np.int64)
    pairs = []

    classes = np.unique(y)
    for c in classes:
        idx = np.flatnonzero(y == c)
        m = idx.size
        if m < 2:
            continue
        max_pairs = m * (m - 1) // 2
        k = int(min(per_class, max_pairs))
        if k <= 0:
            continue

        got = 0
        tries = 0
        while got < k and tries < 10 * k:
            i = int(rng.choice(idx))
            j = int(rng.choice(idx))
            if i == j:
                tries += 1
                continue
            if i > j:
                i, j = j, i
            pairs.append((i, j, 1))
            got += 1
            tries += 1
    return pairs


def _sample_negative_pairs(y: np.ndarray, rng: np.random.Generator, total: int):
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    pairs = []
    if n < 2 or total <= 0:
        return pairs

    tries = 0
    got = 0
    while got < total and tries < 20 * total:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            tries += 1
            continue
        if y[i] == y[j]:
            tries += 1
            continue
        if i > j:
            i, j = j, i
        pairs.append((i, j, 0))
        got += 1
        tries += 1
    return pairs


def _sample_pairs_uniform(n: int, rng: np.random.Generator, n_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Amostra n_pairs pares (i<j) aproximadamente uniformes (com reposição).

    Observações:
      - Para n grande, amostragem SEM reposição/sem duplicatas é cara e desnecessária aqui.
      - Duplicatas têm efeito desprezível nas métricas/ROC quando n_pairs é grande.

    Retorna:
      ii, jj: arrays int64 com shape (n_pairs,), com ii[k] < jj[k] e ii[k] != jj[k].
    """
    n = int(n)
    n_pairs = int(n_pairs)
    if n < 2 or n_pairs <= 0:
        return (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64))

    ii = rng.integers(0, n, size=n_pairs, dtype=np.int64)
    jj = rng.integers(0, n, size=n_pairs, dtype=np.int64)

    # garante i != j (re-amostra apenas onde necessário)
    mask = (ii == jj)
    while np.any(mask):
        jj[mask] = rng.integers(0, n, size=int(np.sum(mask)), dtype=np.int64)
        mask = (ii == jj)

    # normaliza para i < j
    swap = ii > jj
    if np.any(swap):
        tmp = ii[swap].copy()
        ii[swap] = jj[swap]
        jj[swap] = tmp

    return ii, jj


def roc_auc_fast(scores: np.ndarray, truth: np.ndarray) -> float:
    """Calcula AUC ROC binária em O(n log n).

    Motivação: a função roc_curve_simple() existente enumera TODOS os thresholds únicos,
    o que explode para centenas de milhares/milhões de pares. Aqui usamos o método padrão:
    ordena por score decrescente e calcula cumulativos (equivalente ao sklearn).

    Ref (intuição/ROC): T. Fawcett, "An introduction to ROC analysis", Pattern Recognition Letters, 2006.
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    truth = np.asarray(truth).astype(bool).ravel()

    if scores.size == 0:
        return 0.0

    # total de positivos/negativos
    P = int(np.sum(truth))
    N = int(truth.size - P)
    if P == 0 or N == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    scores_s = scores[order]
    truth_s = truth[order].astype(np.int8)

    tp = np.cumsum(truth_s, dtype=np.int64)
    fp = np.cumsum(1 - truth_s, dtype=np.int64)

    # pontos apenas onde o score muda (thresholds distintos)
    distinct = np.where(np.diff(scores_s) != 0)[0]
    idx = np.r_[distinct, truth_s.size - 1]

    tpr = tp[idx] / float(P)
    fpr = fp[idx] / float(N)

    # inclui origem (0,0)
    tpr = np.r_[0.0, tpr.astype(np.float64, copy=False)]
    fpr = np.r_[0.0, fpr.astype(np.float64, copy=False)]

    # integral trapezoidal
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float(np.trapz(tpr, fpr))
    return auc


def auth_macro_auc_per_identity(
        emb: np.ndarray,
        P: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        pmax: np.ndarray,
        rng: np.random.Generator,
        cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Macro ROC-AUC de AUTENTICAÇÃO por identidade (one-vs-all).

    Para cada classe/identidade c:
      - Positivos: pares (i,j) com y[i]==c e y[j]==c (mesma identidade)
      - Negativos: pares (i,j) com y[i]==c e y[j]!=c (ou vice-versa; aqui usamos i em c e j fora)
      - Calcula AUC(c) para diferentes scores de verificação:
          1) cosine  : dot de embeddings L2-normalizados (cosine similarity)
          2) prob_dot: dot entre vetores de probas por classe
          3) id+conf : (mesmo ID predito) * min(conf_i, conf_j)
    No fim, retorna média simples (macro) dos AUC(c) nas classes válidas.

    Observação importante:
      - Isto difere do AUC global (micro-AUC) que usa pares amostrados no dataset todo.
      - Macro-AUC dá o mesmo peso para cada identidade/classe, mesmo se algumas têm muito mais imagens.
    """
    enable = bool(cfg.get("enable", False))
    if not enable:
        return {"enabled": False}

    pos_pairs_per_class = int(cfg.get("pos_pairs_per_class", 30))
    neg_pairs_per_class = int(cfg.get("neg_pairs_per_class", 60))
    classes_fraction = float(cfg.get("classes_fraction", 1.0))
    max_classes = int(cfg.get("max_classes", 0))
    progress_every = int(cfg.get("progress_every", 100))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pmax = np.asarray(pmax, dtype=np.float32).ravel()
    emb = np.asarray(emb, dtype=np.float32)
    P = np.asarray(P, dtype=np.float32)

    classes = np.unique(y_true)
    if classes.size == 0:
        return {"enabled": True, "n_classes_used": 0}

    # amostra subconjunto de classes, se configurado
    if 0.0 < classes_fraction < 1.0:
        k = max(1, int(round(classes.size * classes_fraction)))
        classes = rng.choice(classes, size=k, replace=False)

    if max_classes > 0 and classes.size > max_classes:
        classes = rng.choice(classes, size=max_classes, replace=False)

    auc_cos_list: list[float] = []
    auc_prob_list: list[float] = []
    auc_conf_list: list[float] = []
    used_classes: list[int] = []

    t0 = time.time()
    for t, c in enumerate(classes.tolist(), start=1):
        idx = np.where(y_true == c)[0].astype(np.int64, copy=False)
        if idx.size < 2:
            continue
        idx_other = np.where(y_true != c)[0].astype(np.int64, copy=False)
        if idx_other.size < 1:
            continue

        k_pos = min(pos_pairs_per_class, 10_000_000)  # guard rail
        k_neg = min(neg_pairs_per_class, 10_000_000)
        if k_pos <= 0 or k_neg <= 0:
            continue

        # --- amostra positivos (c vs c)
        ii_pos = idx[rng.integers(0, idx.size, size=k_pos, dtype=np.int64)]
        jj_pos = idx[rng.integers(0, idx.size, size=k_pos, dtype=np.int64)]
        mask = (ii_pos == jj_pos)
        # evita i==j (re-amostra apenas onde necessário)
        while np.any(mask):
            jj_pos[mask] = idx[rng.integers(0, idx.size, size=int(np.sum(mask)), dtype=np.int64)]
            mask = (ii_pos == jj_pos)

        # --- amostra negativos (c vs ~c)
        ii_neg = idx[rng.integers(0, idx.size, size=k_neg, dtype=np.int64)]
        jj_neg = idx_other[rng.integers(0, idx_other.size, size=k_neg, dtype=np.int64)]

        ii = np.concatenate([ii_pos, ii_neg], axis=0)
        jj = np.concatenate([jj_pos, jj_neg], axis=0)
        truth = np.concatenate([
            np.ones((k_pos,), dtype=np.int8),
            np.zeros((k_neg,), dtype=np.int8),
        ], axis=0).astype(bool)

        # scores contínuos
        s_cos = np.sum(emb[ii] * emb[jj], axis=1)
        s_prob = np.sum(P[ii] * P[jj], axis=1)
        same_id = (y_pred[ii] == y_pred[jj]).astype(np.float32)
        conf = np.minimum(pmax[ii], pmax[jj])
        s_conf = conf * same_id

        auc_cos_list.append(float(roc_auc_fast(s_cos, truth)))
        auc_prob_list.append(float(roc_auc_fast(s_prob, truth)))
        auc_conf_list.append(float(roc_auc_fast(s_conf, truth)))
        used_classes.append(int(c))

        if progress_every > 0 and (t % progress_every == 0):
            dt = max(1e-6, time.time() - t0)
            rate = t / dt
            remaining = (len(classes) - t) / max(1e-9, rate)
            print(f"  [AUTH][macro-AUC] classes processadas: {t}/{len(classes)} | usadas={len(used_classes)} | "
                  f"vel={rate:.1f} cls/s | ETA~{remaining:.1f}s")

    if len(used_classes) == 0:
        return {"enabled": True, "n_classes_used": 0}

    return {
        "enabled": True,
        "n_classes_used": int(len(used_classes)),
        "classes_used": used_classes,
        "cos_mean": float(np.mean(auc_cos_list)),
        "prob_mean": float(np.mean(auc_prob_list)),
        "conf_mean": float(np.mean(auc_conf_list)),
        "cos_median": float(np.median(auc_cos_list)),
        "prob_median": float(np.median(auc_prob_list)),
        "conf_median": float(np.median(auc_conf_list)),
    }

# Referência para similaridade por dot/cosseno; avaliação em lote.
# F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet…,” in Proc. IEEE CVPR, 2015.

def _pair_dot_scores_batched(A: np.ndarray, ii: np.ndarray, jj: np.ndarray,
                            batch: int = 50000, label: str = "pairs") -> np.ndarray:
    """Calcula dot(A[ii], A[jj]) em batches com tracker de progresso.

    Usado para autenticação por similaridade (ex.: cosine) e para prob_dot (dot entre vetores de probas).
    """
    A = np.asarray(A, dtype=np.float32)
    ii = np.asarray(ii, dtype=np.int64).ravel()
    jj = np.asarray(jj, dtype=np.int64).ravel()
    n_pairs = int(ii.size)
    out = np.empty((n_pairs,), dtype=np.float32)

    if n_pairs == 0:
        return out

    # Ajuste dinâmico de batch para evitar estouro de memória quando a dimensão é grande
    # (ex.: P tem milhares de classes; fancy-indexing já materializa (batch, dim)).
    batch = int(max(1, batch))
    if A.ndim == 2:
        d = int(A.shape[1])
        max_elems = 25_000_000  # ~100MB por matriz float32
        if d > 0:
            batch = min(batch, max(1, max_elems // d))

    # tracker: imprime ~a cada 10% ou a cada 50k, o que ocorrer primeiro
    step_print = max(50000, (n_pairs // 10) if n_pairs >= 10 else 1)

    done = 0
    for start in range(0, n_pairs, batch):
        end = min(n_pairs, start + batch)
        Ai = A[ii[start:end]]
        Aj = A[jj[start:end]]
        out[start:end] = np.einsum("ij,ij->i", Ai, Aj, optimize=True).astype(np.float32, copy=False)

        done = end
        if done == n_pairs or (done % step_print) == 0:
            pct = 100.0 * done / float(n_pairs)
            print(f"    [AUC-pairs] {label}: {done}/{n_pairs} ({pct:.1f}%)")

    return out

    batch = int(max(1, batch))
    # tracker: imprime ~a cada 10% ou a cada 50k, o que ocorrer primeiro
    step_print = max(50000, (n_pairs // 10) if n_pairs >= 10 else 1)

    done = 0
    for start in range(0, n_pairs, batch):
        end = min(n_pairs, start + batch)
        out[start:end] = np.sum(A[ii[start:end]] * A[jj[start:end]], axis=1).astype(np.float32, copy=False)
        done = end
        if done == n_pairs or (done % step_print) == 0:
            pct = 100.0 * done / float(n_pairs)
            print(f"    [AUC-pairs] {label}: {done}/{n_pairs} ({pct:.1f}%)")
    return out


def _binary_counts_from_pred_truth(pred: np.ndarray, truth: np.ndarray) -> Dict[str, int]:
    pred = np.asarray(pred).astype(bool)
    truth = np.asarray(truth).astype(bool)
    TP = int(np.sum(pred & truth))
    TN = int(np.sum((~pred) & (~truth)))
    FP = int(np.sum(pred & (~truth)))
    FN = int(np.sum((~pred) & truth))
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def _binary_metrics_from_counts(cm: Dict[str, int]) -> Dict[str, float]:
    TP = float(cm["TP"]);
    TN = float(cm["TN"]);
    FP = float(cm["FP"]);
    FN = float(cm["FN"])
    denom = TP + TN + FP + FN
    acc = (TP + TN) / denom if denom > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    tpr = rec
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "balanced_acc": float(bal_acc)}


def _score_summary(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32).ravel()
    if scores.size == 0:
        return {"n": 0}
    qs = np.quantile(scores, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).astype(np.float32)
    return {
        "n": int(scores.size),
        "min": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "median": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

# Implementação das avaliações de autenticação de pares por similaridade/distância com threshold
# F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet” in Proc. IEEE CVPR, 2015.
# T. Fawcett, “An introduction to ROC analysis,” Pattern Recognition Letters, vol. 27, no. 8, pp. 861–874, 2006.

def build_tuning_pairs(y: np.ndarray, rng: np.random.Generator,
                       pos_pairs_per_class: int, neg_pairs_total: int,
                       balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Gera pares (i,j) com rótulo binário de "mesma classe?" (tt=1) ou "classes diferentes?" (tt=0).
    Tuning balanceado: limita #negativos ~ #positivos para evitar "acurácia alta só chutando negativo".
    """
    y = np.asarray(y, dtype=np.int64)
    pos_pairs = _sample_positive_pairs_per_class(y, rng, per_class=int(pos_pairs_per_class))
    n_pos = int(len(pos_pairs))
    n_neg_target = int(neg_pairs_total)
    if balanced and n_pos > 0:
        n_neg_target = min(int(neg_pairs_total), n_pos)
    neg_pairs = _sample_negative_pairs(y, rng, total=int(n_neg_target))
    pairs = pos_pairs + neg_pairs
    if len(pairs) == 0:
        ii = np.zeros((0,), dtype=np.int64)
        jj = np.zeros((0,), dtype=np.int64)
        tt = np.zeros((0,), dtype=np.int8)
        info = {"pairs_used": 0, "pos_pairs": 0, "neg_pairs": 0}
        return ii, jj, tt, info

    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    tt = np.array([p[2] for p in pairs], dtype=np.int8)
    info = {"pairs_used": int(len(pairs)), "pos_pairs": int(len(pos_pairs)), "neg_pairs": int(len(neg_pairs))}
    return ii, jj, tt, info


def tune_threshold_scores(scores: np.ndarray, truth: np.ndarray, q_grid: int,
                          metric: str = "f1") -> Tuple[float, Dict[str, Any]]:
    """
    Escolhe threshold maximizando F1 ou balanced-acc (ou acc).
    - scores: valores contínuos (maior = mais provável ser a mesma classe)
    - truth: 1 (mesma classe) / 0 (diferente)
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    truth = np.asarray(truth, dtype=np.int8).ravel()
    if scores.size == 0:
        return 0.5, {"note": "sem scores para tuning; usando thr=0.5"}

    q_grid = int(max(21, q_grid))
    qs = np.linspace(0.0, 1.0, q_grid, dtype=np.float32)
    thrs = np.unique(np.quantile(scores, qs))
    best_thr = float(thrs[0])
    best_val = -1.0
    best_cm = None
    best_metrics = None

    metric = (metric or "f1").strip().lower()
    if metric not in ("f1", "balanced_acc", "acc"):
        metric = "f1"

    for thr in thrs:
        pred = scores >= float(thr)
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        mets = _binary_metrics_from_counts(cm)
        val = mets["f1"] if metric == "f1" else (mets["balanced_acc"] if metric == "balanced_acc" else mets["acc"])
        if val > best_val:
            best_val = float(val)
            best_thr = float(thr)
            best_cm = cm
            best_metrics = mets

    stats = {
        "metric_optimized": metric,
        "best_metric_value": float(best_val),
        "best_thr": float(best_thr),
        "best_cm": best_cm,
        "best_metrics": best_metrics,
        "thr_candidates": int(thrs.size),
        "scores_summary": _score_summary(scores),
        "pos_scores_summary": _score_summary(scores[truth == 1]),
        "neg_scores_summary": _score_summary(scores[truth == 0]),
    }
    return best_thr, stats


def tune_threshold_identity_confidence(y_pred: np.ndarray, pmax: np.ndarray,
                                       ii: np.ndarray, jj: np.ndarray, truth: np.ndarray,
                                       q_grid: int, metric: str = "f1") -> Tuple[float, Dict[str, Any]]:
    """
    Regra de verificação baseada em identidade prevista e confiança:
      same = (y_pred[i] == y_pred[j]) AND (min(pmax_i, pmax_j) >= thr_conf)

    Faz tuning do thr_conf (balanceado via truth fornecido).
    """
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pmax = np.asarray(pmax, dtype=np.float32)
    ii = np.asarray(ii, dtype=np.int64)
    jj = np.asarray(jj, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int8).ravel()
    if ii.size == 0:
        return 0.5, {"note": "sem pares para tuning; usando thr=0.5"}

    same_id = (y_pred[ii] == y_pred[jj])
    conf = np.minimum(pmax[ii], pmax[jj]).astype(np.float32, copy=False)

    # thresholds em [0,1] por quantis do conf (mas inclui 0 e 1)
    q_grid = int(max(21, q_grid))
    qs = np.linspace(0.0, 1.0, q_grid, dtype=np.float32)
    thrs = np.unique(np.quantile(conf, qs))
    best_thr = float(thrs[0])
    best_val = -1.0
    best_cm = None
    best_metrics = None

    metric = (metric or "f1").strip().lower()
    if metric not in ("f1", "balanced_acc", "acc"):
        metric = "f1"

    for thr in thrs:
        pred = same_id & (conf >= float(thr))
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        mets = _binary_metrics_from_counts(cm)
        val = mets["f1"] if metric == "f1" else (mets["balanced_acc"] if metric == "balanced_acc" else mets["acc"])
        if val > best_val:
            best_val = float(val)
            best_thr = float(thr)
            best_cm = cm
            best_metrics = mets

    stats = {
        "metric_optimized": metric,
        "best_metric_value": float(best_val),
        "best_thr": float(best_thr),
        "best_cm": best_cm,
        "best_metrics": best_metrics,
        "thr_candidates": int(thrs.size),
        "conf_summary_all": _score_summary(conf),
        "conf_summary_sameid": _score_summary(conf[same_id]),
        "conf_summary_diffid": _score_summary(conf[~same_id]),
    }
    return best_thr, stats


def eval_auth_pairs_full_prob(P: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    """Avalia todas as combinações com score = dot(P_i, P_j)."""
    P = np.asarray(P, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        sims = P[i + 1:] @ P[i]  # (n-i-1,)
        pred = sims >= float(thr)
        truth = (y[i + 1:] == y[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    acc = (TP + TN) / pairs if pairs > 0 else 0.0
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            **{f"m_{k}": v for k, v in mets.items()}}


def eval_auth_pairs_full_identity_confidence(y_pred: np.ndarray, pmax: np.ndarray, y_true: np.ndarray,
                                             thr_conf: float) -> Dict[str, Any]:
    """Avalia todas as combinações com regra: same_id_pred & min_conf >= thr_conf."""
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pmax = np.asarray(pmax, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)
    n = int(y_true.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        same_id = (y_pred[i + 1:] == y_pred[i])
        conf = np.minimum(pmax[i + 1:], pmax[i])
        pred = same_id & (conf >= float(thr_conf))
        truth = (y_true[i + 1:] == y_true[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    acc = (TP + TN) / pairs if pairs > 0 else 0.0
    mets = _binary_metrics_from_counts({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
    return {"mode": "full", "pairs": int(pairs), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            **{f"m_{k}": v for k, v in mets.items()}}


def roc_curve_simple(scores: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ROC simples (sem sklearn). Retorna fpr, tpr, thresholds, auc.
    truth: 1/0
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    truth = np.asarray(truth, dtype=np.int8).ravel()
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.0
    thrs = np.unique(scores)[::-1]
    # adiciona um thr > max para ponto (0,0)
    thrs = np.concatenate(([float(np.max(scores) + 1.0)], thrs, [float(np.min(scores) - 1.0)])).astype(np.float32)
    Ppos = float(np.sum(truth == 1))
    Nneg = float(np.sum(truth == 0))
    tpr = []
    fpr = []
    for thr in thrs:
        pred = scores >= float(thr)
        cm = _binary_counts_from_pred_truth(pred, truth == 1)
        TP = cm["TP"];
        FP = cm["FP"]
        tpr.append(TP / Ppos if Ppos > 0 else 0.0)
        fpr.append(FP / Nneg if Nneg > 0 else 0.0)
    fpr = np.asarray(fpr, dtype=np.float32)
    tpr = np.asarray(tpr, dtype=np.float32)
    # ordena por fpr crescente
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = float(np.trapezoid(tpr, fpr)) if (hasattr(np, "trapezoid") and fpr.size > 1) else (float(np.trapz(tpr, fpr)) if fpr.size > 1 else 0.0)
    return fpr, tpr, thrs.astype(np.float32), auc


def plot_roc_ova_for_classes(pick_classes: List[int], y_true: np.ndarray, proba: np.ndarray,
                             classes_modelo: np.ndarray, out_dir: Path, roc_cfg: Dict[str, Any]):
    """Plota ROC one-vs-all (score = probabilidade da classe)."""
    if not roc_cfg or not bool(roc_cfg.get("enable", True)):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ROC] Falha ao importar matplotlib ({e}). Pulando ROC.")
        return

    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float32)
    classes_modelo = np.asarray(classes_modelo, dtype=np.int64)
    pick_classes = [int(c) for c in pick_classes]

    plt.figure()
    any_curve = False
    for c in pick_classes:
        idx = np.where(classes_modelo == int(c))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        scores = proba[:, k]
        truth = (y_true == int(c)).astype(np.int8)
        fpr, tpr, _thrs, auc = roc_curve_simple(scores, truth)
        if fpr.size == 0:
            continue
        any_curve = True
        plt.plot(fpr, tpr, label=f"c={c} AUC={auc:.3f}")

    if not any_curve:
        print("[ROC] Sem curvas para plotar.")
        return

    plt.plot([0, 1], [0, 1], linestyle="--", label="aleatório")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC One-vs-All (10 classes)")
    plt.legend(loc="lower right")

    if bool(roc_cfg.get("save_png", True)):
        png_name = str(roc_cfg.get("png_name", "roc_ova_10classes.png"))
        out_path = out_dir / png_name
        try:
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"[ROC] PNG salvo em: {out_path}")
        except Exception as e:
            print(f"[ROC] Falha ao salvar PNG ({e}).")

    if bool(roc_cfg.get("show_plots", True)):
        plt.show()
    else:
        plt.close()


def eval_auth_pairs_full(emb: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    """
    Avalia todas as combinações (i<j) sem armazenar índices enormes.
    Retorna contagens TP/TN/FP/FN e acc.
    """
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2:
        return {"note": "N<2", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    for i in range(n - 1):
        sims = emb[i + 1:] @ emb[i]  # (n-i-1,)
        pred = sims >= float(thr)
        truth = (y[i + 1:] == y[i])
        TP += int(np.sum(pred & truth))
        TN += int(np.sum((~pred) & (~truth)))
        FP += int(np.sum(pred & (~truth)))
        FN += int(np.sum((~pred) & truth))

    pairs = n * (n - 1) // 2
    acc = (TP + TN) / pairs if pairs > 0 else 0.0
    return {"mode": "full", "pairs": int(pairs), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def eval_auth_pairs_sample(emb: np.ndarray, y: np.ndarray, thr: float, rng: np.random.Generator, n_pairs: int):
    emb = np.asarray(emb, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = int(y.size)
    if n < 2 or n_pairs <= 0:
        return {"note": "N<2 ou n_pairs<=0", "pairs": 0, "acc": 0.0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}

    TP = TN = FP = FN = 0
    got = 0
    tries = 0
    while got < n_pairs and tries < 10 * n_pairs:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            tries += 1
            continue
        if i > j:
            i, j = j, i
        sim = float(np.dot(emb[i], emb[j]))
        pred = sim >= float(thr)
        truth = (y[i] == y[j])
        if pred and truth:
            TP += 1
        elif (not pred) and (not truth):
            TN += 1
        elif pred and (not truth):
            FP += 1
        else:
            FN += 1
        got += 1
        tries += 1

    acc = (TP + TN) / got if got > 0 else 0.0
    return {"mode": "sample", "pairs": int(got), "acc": float(acc), "TP": TP, "TN": TN, "FP": FP, "FN": FN}


# ============================================================
# Split / reconstrução de avaliação
# ============================================================

# Referências de separação de sets sem leakage
# P. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” in Proc. IJCAI, 1995.
# # S. Kaufman, S. Rosset, and C. Perlich, “Leakage in Data Mining: Formulation, Detection, and Avoidance,” in Proc. ACM KDD, 2012.

def _train_test_split_with_meta(X: np.ndarray, y: np.ndarray,
                                paths: Optional[np.ndarray], ids: Optional[np.ndarray],
                                test_size: float, seed: int):
    idx = np.arange(y.size, dtype=np.int64)
    idx_tr, idx_te = train_test_split(
        idx,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )

    def take(a, idx_):
        return a[idx_] if a is not None else None

    return (X[idx_tr], X[idx_te],
            y[idx_tr], y[idx_te],
            take(paths, idx_tr), take(paths, idx_te),
            take(ids, idx_tr), take(ids, idx_te))


def build_eval_split_from_payload(X: np.ndarray, y: np.ndarray,
                                  paths: Optional[np.ndarray],
                                  ids: Optional[np.ndarray],
                                  payload_conf: Dict[str, Any],
                                  standardizer: Dict[str, Any],
                                  classes_modelo: np.ndarray):
    idx_global = np.arange(y.size, dtype=np.int64)

    classes_elig = selecionar_classes_elegiveis(y, int(payload_conf["min_amostras_por_classe"]))
    classes_sel = amostrar_classes(classes_elig, float(payload_conf["frac_classes"]), int(payload_conf["seed_classes"]))
    X1, y1, p1, id1, _ = filtrar_por_classes(X, y, classes_sel, paths=paths, ids=ids, idx_global=idx_global)

    # (anti-leakage) Se o dataset de entrada já é o HOLDOUT FINAL salvo pelo treino
    # (final_test_manifest.csv), não criamos um novo holdout interno aqui.
    # Usamos X1/y1 como TESTE e apenas filtramos classes com amostras suficientes
    # para a lógica de autenticação (pares / k-folds).
    if bool(payload_conf.get("explicit_holdout_manifest", False)):
        min_train = int(max(int(payload_conf["min_train_por_classe"]), int(payload_conf["k_folds"])))
        cls, cts = np.unique(y1, return_counts=True)
        cls_keep = cls[cts >= min_train]

        X_test, y_test, p_test, id_test, _ = filtrar_por_classes(
            X1, y1, cls_keep, paths=p1, ids=id1, idx_global=None
        )

        mean = np.asarray(standardizer["mean"], dtype=np.float32)
        std = np.asarray(standardizer["std"], dtype=np.float32)
        X_test_feat = apply_standardizer(X_test, mean, std)

        # manter compatibilidade com o modelo final
        X_test_final, y_test_final, p_test_final, id_test_final, _ = filtrar_por_classes(
            X_test_feat, y_test, classes_modelo,
            paths=p_test, ids=id_test, idx_global=None
        )
        return X_test_final, y_test_final, p_test_final, id_test_final


    X_train_all, X_test_all, y_train_all, y_test_all, p_train_all, p_test_all, id_train_all, id_test_all = \
        _train_test_split_with_meta(
            X1, y1, p1, id1,
            test_size=float(payload_conf["test_frac"]),
            seed=int(payload_conf["seed_split"]),
        )

    min_train = int(max(int(payload_conf["min_train_por_classe"]), int(payload_conf["k_folds"])))
    cls, cts = np.unique(y_train_all, return_counts=True)
    cls_keep = cls[cts >= min_train]

    X_train, y_train, p_train, id_train, _ = filtrar_por_classes(X_train_all, y_train_all, cls_keep,
                                                                 paths=p_train_all, ids=id_train_all,
                                                                 idx_global=None)
    cls_train = np.unique(y_train).astype(np.int64, copy=False)
    X_test, y_test, p_test, id_test, _ = filtrar_por_classes(X_test_all, y_test_all, cls_train,
                                                             paths=p_test_all, ids=id_test_all,
                                                             idx_global=None)

    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    X_test_feat = apply_standardizer(X_test, mean, std)

    X_test_final, y_test_final, p_test_final, id_test_final, _ = filtrar_por_classes(
        X_test_feat, y_test, classes_modelo,
        paths=p_test, ids=id_test, idx_global=None
    )
    return X_test_final, y_test_final, p_test_final, id_test_final


def main():
    # RNG para amostragens reprodutíveis (one-vs-all, pares, etc.)
    rng = np.random.default_rng(int(SCRIPT_CONFIG.get("seed", 42)))

    # ------------------------------------------------------------
    # Resolução do run_dir (para salvar outputs junto do treino)
    # ------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    model_payload_file = str(SCRIPT_CONFIG.get("model_payload_file", "models/final_model_best.keras"))

    def _detect_latest_run_dir(base: Path) -> Optional[Path]:
        candidates: List[Path] = []

        # 1) Se o script está dentro do próprio run_dir
        if (base / "hyperparameters.json").exists() and (base / model_payload_file).exists():
            candidates.append(base)

        # 2) Padrão do projeto: <dataset_dir>/runs/<timestamp>/
        # Procuramos por qualquer .../runs/<run>/models/final_model_best.keras nas redondezas.
        search_roots = [base] + list(base.parents)[:4]
        for root in search_roots:
            for runs_dir in root.glob("**/runs"):
                if not runs_dir.is_dir():
                    continue
                for run_dir in runs_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    if (run_dir / model_payload_file).exists() and (run_dir / "hyperparameters.json").exists():
                        candidates.append(run_dir)

        if not candidates:
            return None

        # Preferir o nome (timestamps) e, como desempate, mtime
        def _key(p: Path):
            try:
                name_key = p.name
            except Exception:
                name_key = ""
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            return (name_key, mtime)

        return sorted(candidates, key=_key)[-1]

    run_dir_override = SCRIPT_CONFIG.get("run_dir_override", None)
    if run_dir_override:
        out_dir = Path(run_dir_override).resolve()
    else:
        detected = _detect_latest_run_dir(script_dir)
        out_dir = detected if detected is not None else script_dir

    # ------------------------------------------------------------
    # Carrega modelo (CNN) e configurações do treino (hyperparams)
    # ------------------------------------------------------------
    model_path = out_dir / model_payload_file
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    try:
        modelo = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception:
        # fallback sem compile=False (algumas versões)
        modelo = tf.keras.models.load_model(str(model_path))

    print(f"[INFO] Modelo CNN carregado: {model_path}")

    cfg_train_path = out_dir / "hyperparameters.json"
    if not cfg_train_path.exists():
        raise FileNotFoundError(f"hyperparameters.json não encontrado: {cfg_train_path}")
    with open(cfg_train_path, "r", encoding="utf-8") as f:
        cfg_train = json.load(f)

    # label2idx (map label original -> y reindexado compatível com o modelo)
    label2idx_path = out_dir / "label2idx.json"
    label2idx: Dict[int, int] = {}
    if label2idx_path.exists():
        with open(label2idx_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        label2idx = {int(k): int(v) for k, v in raw.items()}

    # número de classes (preferimos hyperparams; fallback: shape do modelo)
    num_classes = int(cfg_train.get("NUM_CLASSES", 0) or (modelo.output_shape[-1] if hasattr(modelo, "output_shape") else 0))
    if num_classes <= 0:
        raise ValueError("Não foi possível determinar NUM_CLASSES (hyperparameters.json ou output_shape do modelo).")

    classes_modelo = np.arange(num_classes, dtype=np.int64)
    print(f"[INFO] Num classes: {num_classes}")

    # ------------------------------------------------------------
    # Dataset (manifest.csv) – localizar caminho de forma robusta
    # ------------------------------------------------------------
    dataset_dir_cfg = str(cfg_train.get("DATASET_DIR", "")).strip()
    dataset_dir = Path(dataset_dir_cfg) if dataset_dir_cfg else None
    if dataset_dir is None or not dataset_dir.exists():
        # padrão: .../<dataset_dir>/runs/<run_dir>/  -> dataset_dir = run_dir/../..
        if out_dir.parent.name == "runs":
            dataset_dir = out_dir.parent.parent
        else:
            dataset_dir = out_dir

    dataset_path = SCRIPT_CONFIG.get("dataset_path_override", None)
    if dataset_path:
        dataset_path = Path(dataset_path).resolve()
    else:
        dataset_path = (dataset_dir / "manifest.csv").resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"manifest.csv não encontrado: {dataset_path}")

    # ------------------------------------------------------------
    # (NOVO) Anti-leakage: se o script de treino salvou o holdout final
    #        (final_test_manifest.csv) no run_dir, reutilizamos aqui.
    #        Opcionalmente, usamos cv_manifest.csv para o tuning.
    # ------------------------------------------------------------
    holdout_path = (out_dir / "final_test_manifest.csv").resolve()
    cv_pool_path = (out_dir / "cv_manifest.csv").resolve()

    dataset_path_eval = dataset_path
    dataset_path_tune = dataset_path
    explicit_holdout = False

    if holdout_path.exists():
        dataset_path_eval = holdout_path
        explicit_holdout = True
        if cv_pool_path.exists():
            dataset_path_tune = cv_pool_path
        else:
            dataset_path_tune = dataset_path_eval
            print("[WARN] cv_manifest.csv não encontrado; tuning será feito a partir do próprio holdout (menos ideal).")

        print(f"[INFO] (anti-leakage) dataset_eval = {dataset_path_eval}")
        print(f"[INFO] (anti-leakage) dataset_tune = {dataset_path_tune}")
    else:
        print(f"[INFO] Dataset (manifest) = {dataset_path}")

    # ------------------------------------------------------------
    # Parâmetros (mantendo a lógica do script original)
    # ------------------------------------------------------------
    seed_split = int(cfg_train.get("SEED", 42))
    k_folds = int(cfg_train.get("K_FOLDS", 2))

    # No treino, o holdout final é por-classe; aqui usamos como test_frac (reconstrução)
    test_frac = float(cfg_train.get("FINAL_TEST_FRACTION_PER_CLASS", 0.2))

    conf = {
        "dataset_path": str(dataset_path_eval),
        "explicit_holdout_manifest": bool(explicit_holdout),
        "test_frac": test_frac,
        "seed_split": seed_split,
        "min_amostras_por_classe": k_folds,  # mesma ideia do original: evitar classes muito pequenas
        "frac_classes": float(SCRIPT_CONFIG.get("frac_classes", 1.0)),
        "seed_classes": int(cfg_train.get("SEED", 42) + 17),
        "min_train_por_classe": k_folds,
        "k_folds": k_folds,
    }

    # Normalização usada no treino (ImageNet por padrão)
    norm_mean = np.asarray(cfg_train.get("NORM_MEAN", [0.485, 0.456, 0.406]), dtype=np.float32)
    norm_std = np.asarray(cfg_train.get("NORM_STD", [0.229, 0.224, 0.225]), dtype=np.float32)
    standardizer = {"mean": norm_mean, "std": norm_std}

    inference_params = {
        "batch_size": int(cfg_train.get("BATCH_SIZE", 64)),
        "embedding_layer_name": str(SCRIPT_CONFIG.get("embedding_layer_name", "dense")),
        "img_size": int(cfg_train.get("IMG_SIZE", 128)),
        "channels": int(cfg_train.get("IN_CHANNELS", 3)),
        "assume_ingested_size": bool(cfg_train.get("ASSUME_INGESTED_SIZE", True)),
        "jpeg_dct_method": str(cfg_train.get("JPEG_DCT_METHOD", "INTEGER_FAST")),
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
    }
    # ------------------------------------------------------------
    # Carregar dataset e construir splits (eval + tune) como no original
    # ------------------------------------------------------------
    # (anti-leakage) Base de AVALIAÇÃO
    X_eval_base, y_eval_base, meta_eval = carregar_dataset_joblib(str(dataset_path_eval), label2idx=label2idx)

    paths_base = meta_eval.get("paths", None)
    ids_base = meta_eval.get("ids", None)

    if paths_base is not None:
        paths_base = np.asarray(paths_base)
    if ids_base is not None:
        ids_base = np.asarray(ids_base)

    # (anti-leakage) Base de TUNING (padrão: manifest completo; se existir cv_manifest.csv, usa o pool de CV)
    X, y, meta = carregar_dataset_joblib(str(dataset_path_tune), label2idx=label2idx)

    paths = meta.get("paths", None)
    ids = meta.get("ids", None)

    if paths is not None:
        paths = np.asarray(paths)
    if ids is not None:
        ids = np.asarray(ids)

    X_eval, y_eval, paths_eval, ids_eval = build_eval_split_from_payload(
        X=X_eval_base,
        y=y_eval_base,
        paths=paths_base,
        ids=ids_base,
        payload_conf=conf,
        standardizer=standardizer,
        classes_modelo=classes_modelo,
    )

    # Reconstrução do conjunto de "tuning" (mesma lógica do original)
    idx_global = np.arange(y.size, dtype=np.int64)

    classes_elig = selecionar_classes_elegiveis(y, conf["min_amostras_por_classe"])
    classes_sel = amostrar_classes(classes_elig, conf["frac_classes"], conf["seed_classes"])

    X1, y1, p1, id1, idx1 = filtrar_por_classes(X, y, classes_sel, paths, ids, idx_global)

    idx_train_all, idx_test_all = train_test_split(
        idx1,
        test_size=conf["test_frac"],
        random_state=conf["seed_split"],
        stratify=y1,
    )

    mask_train_all = np.isin(idx1, idx_train_all)
    X_train_all = X1[mask_train_all]
    y_train_all = y1[mask_train_all]
    p_train_all = p1[mask_train_all] if p1 is not None else None
    id_train_all = id1[mask_train_all] if id1 is not None else None

    cls, cts = np.unique(y_train_all, return_counts=True)
    cls_keep = cls[cts >= conf["min_train_por_classe"]]
    X_train_t, y_train_t, p_train_t, id_train_t, _ = filtrar_por_classes(
        X_train_all, y_train_all, cls_keep, p_train_all, id_train_all, None
    )

    X_train_feat_t = apply_standardizer(X_train_t, norm_mean, norm_std)
    X_tune, y_tune, paths_tune, ids_tune, _ = filtrar_por_classes(
        X_train_feat_t, y_train_t, classes_modelo, p_train_t, id_train_t, None
    )

    # Evitar 'padronização' dupla: se X_tune é caminho, apply_standardizer devolve o próprio X.
    # Para o caso CNN, a normalização acontece em mlp_forward_inference via tf.data (usando inference_params).
    # =========================
    # IDENTIFICAÇÃO (multiclasse)
    # =========================
    # Objetivo: prever o ID (classe) de cada imagem.
    # - Entrada: X_eval (features HOG previamente extraídas).
    # - Saída do modelo: P (distribuição de probabilidade por classe) e y_pred (argmax).
    # - Métrica principal: acurácia top-1; complementos: confusão, e AUC macro (one-vs-rest).
    #
    # Referências (fundamentos):
    # - C. Bishop, *Pattern Recognition and Machine Learning* (classificação multiclasse).
    # - T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*.
    # - T. Fawcett, "An introduction to ROC analysis" (ROC/AUC), 2006.
    y_pred, P = predict_labels(X_eval, classes_modelo, modelo, inference_params)
    acc = accuracy(y_eval, y_pred)
    print("\n[IDENTIFICAÇÃO] Acurácia no conjunto de avaliação:")
    print(f"  acc={acc:.4f} ({int(np.sum(y_eval == y_pred))}/{int(y_eval.size)})")

    # one-vs-all
    n_ova = int(SCRIPT_CONFIG["one_vs_all_n_classes"])
    classes_present = np.unique(y_eval).astype(np.int64, copy=False)
    n_pick = min(n_ova, int(classes_present.size))
    pick = rng.choice(classes_present, size=n_pick, replace=False)

    print(f"\n[IDENTIFICAÇÃO] One-vs-all ({n_pick} classes aleatórias): classes={pick.tolist()}")
    for c in pick.tolist():
        cm = confusion_one_vs_all(y_eval, y_pred, int(c))
        print_confusion_binary(title=f"[One-vs-all] classe={int(c)}", cm=cm)

    # ROC (one-vs-all) para as mesmas 10 classes escolhidas acima (thresholds variados)
    plot_roc_ova_for_classes(
        pick_classes=pick.tolist(),
        y_true=y_eval,
        proba=P,
        classes_modelo=classes_modelo,
        out_dir=out_dir,
        roc_cfg=SCRIPT_CONFIG.get("roc", {}) or {}
    )

    # =========================
    # AUTENTICAÇÃO (verificação: same vs different)
    # =========================
    # Objetivo: dado um par de imagens, decidir se pertencem ao MESMO ID.
    # Estratégia: transformar cada imagem em uma representação e medir similaridade:
    # - cosine em embeddings (vetores contínuos do penúltimo layer / representação interna)
    # - prob_dot: dot entre distribuições P (afinidade entre posteriors)
    # - id+conf: regra discreta (mesmo y_pred) + confiança (min(pmax)) >= threshold
    #
    # Implementação:
    # 1) Extraímos embeddings e (opcionalmente) normalizamos para usar dot ≡ cosine.
    # 2) Fazemos tuning de thresholds em um conjunto separado (y_tune).
    # 3) Avaliamos em pares. Para N grande, amostramos pares para evitar O(N^2).
    # 4) Calculamos AUC ROC em pares com algoritmo O(n log n) (roc_auc_fast), pois
    #    enumerar todos os thresholds únicos não escala para centenas de milhares de pares.
    #
    # Referências (contexto de verificação por embeddings):
    # - Schroff et al., "FaceNet" (embeddings + cosine para verificação), 2015.
    # - Deng et al., "ArcFace" / Wang et al., "CosFace" (margens em espaço angular), 2018.
    # - T. Fawcett, 2006 (ROC/AUC).
    print("\n[AUTH] Extraindo embeddings...")
    emb = extract_embeddings(X_eval, modelo, inference_params)
    pmax = np.max(P, axis=1).astype(np.float32, copy=False)

    auth_cfg = SCRIPT_CONFIG.get("auth", {}) or {}
    tune_metric = str(auth_cfg.get("tune_metric", "f1")).strip().lower()
    if tune_metric not in ("f1", "balanced_acc", "acc"):
        tune_metric = "f1"

    print("[AUTH] Tuning balanceado de thresholds ...")
    # (anti-leakage) o tuning de thresholds deve usar TREINO (X_tune/y_tune), não o TESTE (X_eval/y_eval)
    y_pred_tune, P_tune = predict_labels(X_tune, classes_modelo, modelo, inference_params)
    emb_tune = extract_embeddings(X_tune, modelo, inference_params)
    pmax_tune = np.max(P_tune, axis=1).astype(np.float32, copy=False)

    ii_t, jj_t, tt_t, pair_info = build_tuning_pairs(
        y_tune, rng=rng,
        pos_pairs_per_class=int(auth_cfg.get("pos_pairs_per_class", 50)),
        neg_pairs_total=int(auth_cfg.get("neg_pairs_total", 20000)),
        balanced=True
    )

    if int(pair_info.get("pairs_used", 0)) == 0:
        print("[AUTH] Sem pares suficientes para tuning. Usando thresholds padrão.")
        thr_cos = 0.5
        thr_prob = 0.5
        thr_conf = 0.5
    else:
        sims_cos = np.sum(emb_tune[ii_t] * emb_tune[jj_t], axis=1).astype(np.float32, copy=False)
        sims_prob = np.sum(P_tune[ii_t] * P_tune[jj_t], axis=1).astype(np.float32, copy=False)

        pos_mask = (tt_t == 1)
        neg_mask = (tt_t == 0)
        print(f"  pares tuning: {pair_info} | métrica otimizada: {tune_metric}")

        # print sim_pos vs sim_neg
        print("\n  [SCORES] cosine: pos vs neg")
        print("    pos:", _score_summary(sims_cos[pos_mask]))
        print("    neg:", _score_summary(sims_cos[neg_mask]))
        print("\n  [SCORES] prob_dot: pos vs neg")
        print("    pos:", _score_summary(sims_prob[pos_mask]))
        print("    neg:", _score_summary(sims_prob[neg_mask]))

        thr_cos, stats_cos = tune_threshold_scores(
            sims_cos, tt_t, q_grid=int(auth_cfg.get("threshold_grid_q", 401)), metric=tune_metric
        )
        thr_prob, stats_prob = tune_threshold_scores(
            sims_prob, tt_t, q_grid=int(auth_cfg.get("threshold_grid_q", 401)), metric=tune_metric
        )
        thr_conf, stats_conf = tune_threshold_identity_confidence(
            y_pred=y_pred_tune, pmax=pmax_tune,
            ii=ii_t, jj=jj_t, truth=tt_t,
            q_grid=int(auth_cfg.get("threshold_grid_q", 401)), metric=tune_metric
        )

        print("\n  [TUNING] cosine:", {k: stats_cos.get(k) for k in (
        "metric_optimized", "best_metric_value", "best_thr", "best_cm", "best_metrics")})
        print("  [TUNING] prob_dot:", {k: stats_prob.get(k) for k in (
        "metric_optimized", "best_metric_value", "best_thr", "best_cm", "best_metrics")})
        print("  [TUNING] id+conf:", {k: stats_conf.get(k) for k in
                                      ("metric_optimized", "best_metric_value", "best_thr", "best_cm", "best_metrics")})

    eval_mode = str(auth_cfg.get("eval_mode", "auto")).lower()
    full_if_n_leq = int(auth_cfg.get("full_if_n_leq", 2500))
    n = int(y_eval.size)
    use_full = (eval_mode == "full") or (eval_mode == "auto" and n <= full_if_n_leq)
    print(f"\n[AUTH] Avaliação em pares: mode={'full' if use_full else 'sample'} | N={n}")

    if use_full:
        res_cos = eval_auth_pairs_full(emb, y_eval, thr=float(thr_cos))
        res_prob = eval_auth_pairs_full_prob(P, y_eval, thr=float(thr_prob))
        res_conf = eval_auth_pairs_full_identity_confidence(y_pred=y_pred, pmax=pmax, y_true=y_eval,
                                                            thr_conf=float(thr_conf))

        # AUC ROC em pares (amostrados) mesmo em modo "full"
        # Motivo: AUC exige scores contínuos; calcular ROC/AUC em TODOS os pares O(N^2) pode ficar pesado.
        # Aqui usamos a mesma amostragem controlada por `sample_pairs_if_large` * `pairs_fraction`.
        base_pairs = int(auth_cfg.get("sample_pairs_if_large", 300000))
        frac_pairs = float(auth_cfg.get("pairs_fraction", 1.0))
        frac_pairs = max(0.0, min(1.0, frac_pairs))
        n_pairs_auc = int(round(base_pairs * frac_pairs))
        if base_pairs > 0 and n_pairs_auc < 1:
            n_pairs_auc = 1

        if n_pairs_auc > 0:
            ii_auc, jj_auc = _sample_pairs_uniform(n=int(n), rng=rng, n_pairs=int(n_pairs_auc))
            truth_auc = (y_eval[ii_auc] == y_eval[jj_auc]).astype(np.int8)

            sims_cos_auc = _pair_dot_scores_batched(emb, ii_auc, jj_auc, batch=50000, label="cosine(AUC)")
            sims_prob_auc = _pair_dot_scores_batched(P, ii_auc, jj_auc, batch=50000, label="prob_dot(AUC)")

            same_id_auc = (y_pred[ii_auc] == y_pred[jj_auc])
            conf_auc = np.minimum(pmax[ii_auc], pmax[jj_auc]).astype(np.float32, copy=False)
            score_conf_auc = (conf_auc * same_id_auc.astype(np.float32)).astype(np.float32, copy=False)

            print("  [AUTH] ROC-AUC (autenticação): ordenando cosine(AUC)...")
            auc_cos = roc_auc_fast(sims_cos_auc, truth_auc == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando prob_dot(AUC)...")
            auc_prob = roc_auc_fast(sims_prob_auc, truth_auc == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando id+conf(AUC)...")
            auc_conf = roc_auc_fast(score_conf_auc, truth_auc == 1)

            print(f"  [AUTH] ROC-AUC (autenticação, same vs different, threshold-var; pares amostrados, n={int(ii_auc.size)}): "
                  f"cosine={auc_cos:.4f} | prob_dot={auc_prob:.4f} | id+conf={auc_conf:.4f}")
    else:
        # --- Amostragem de pares para N grande ---
        # Em vez de avaliar todos os pares O(N^2), amostramos uma quantidade configurável.
        # A nova flag `pairs_fraction` permite reduzir rapidamente o custo para prototipagem
        # (ex.: 0.2 usa 20% de `sample_pairs_if_large`).
        base_pairs = int(auth_cfg.get("sample_pairs_if_large", 300000))
        frac_pairs = float(auth_cfg.get("pairs_fraction", 1.0))
        frac_pairs = max(0.0, min(1.0, frac_pairs))
        n_pairs = int(round(base_pairs * frac_pairs))
        if base_pairs > 0 and n_pairs < 1:
            n_pairs = 1

        if n_pairs <= 0:
            res_cos = {"mode": "sample", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
            res_prob = {"mode": "sample", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
            res_conf = {"mode": "sample", "pairs": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
        else:
            ii_s, jj_s = _sample_pairs_uniform(n=int(n), rng=rng, n_pairs=int(n_pairs))
            truth_s = (y_eval[ii_s] == y_eval[jj_s]).astype(np.int8)

            # cosine similarity (dot em embeddings L2-normalizados ≡ cosine)
            sims_cos = _pair_dot_scores_batched(emb, ii_s, jj_s, batch=50000, label="cosine")
            pred_cos = sims_cos >= float(thr_cos)
            cm_cos_s = _binary_counts_from_pred_truth(pred_cos, truth_s == 1)
            mets_cos_s = _binary_metrics_from_counts(cm_cos_s)
            res_cos = {"mode": "sample", "pairs": int(ii_s.size),
                       "TP": cm_cos_s["TP"], "TN": cm_cos_s["TN"], "FP": cm_cos_s["FP"], "FN": cm_cos_s["FN"],
                       **{f"m_{k}": v for k, v in mets_cos_s.items()}}

            # prob_dot: dot entre vetores de probas por classe (interpretação: "afinidade" de distribuição)
            sims_prob = _pair_dot_scores_batched(P, ii_s, jj_s, batch=50000, label="prob_dot")
            pred_prob = sims_prob >= float(thr_prob)
            cm_prob = _binary_counts_from_pred_truth(pred_prob, truth_s == 1)
            mets_prob = _binary_metrics_from_counts(cm_prob)
            res_prob = {"mode": "sample", "pairs": int(ii_s.size),
                        "TP": cm_prob["TP"], "TN": cm_prob["TN"], "FP": cm_prob["FP"], "FN": cm_prob["FN"],
                        **{f"m_{k}": v for k, v in mets_prob.items()}}

            # id+conf: mesmo ID predito E confiança conjunta >= thr_conf
            same_id = (y_pred[ii_s] == y_pred[jj_s])
            conf_s = np.minimum(pmax[ii_s], pmax[jj_s]).astype(np.float32, copy=False)
            pred_conf = same_id & (conf_s >= float(thr_conf))
            cm_conf = _binary_counts_from_pred_truth(pred_conf, truth_s == 1)
            mets_conf = _binary_metrics_from_counts(cm_conf)
            res_conf = {"mode": "sample", "pairs": int(ii_s.size),
                        "TP": cm_conf["TP"], "TN": cm_conf["TN"], "FP": cm_conf["FP"], "FN": cm_conf["FN"],
                        **{f"m_{k}": v for k, v in mets_conf.items()}}

            # AUC ROC em pares (com tracker de progresso no cálculo de scores acima)
            # Atenção: para ROC, usamos scores contínuos (não binarizados pelo threshold).
            score_conf = (conf_s * same_id.astype(np.float32)).astype(np.float32, copy=False)
            print("  [AUTH] ROC-AUC (autenticação): ordenando cosine...")
            auc_cos = roc_auc_fast(sims_cos, truth_s == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando prob_dot...")
            auc_prob = roc_auc_fast(sims_prob, truth_s == 1)
            print("  [AUTH] ROC-AUC (autenticação): ordenando id+conf...")
            auc_conf = roc_auc_fast(score_conf, truth_s == 1)
            print(f"  [AUTH] ROC-AUC (autenticação, same vs different, threshold-var; pares amostrados, n={int(ii_s.size)}): "
                  f"cosine={auc_cos:.4f} | prob_dot={auc_prob:.4f} | id+conf={auc_conf:.4f}")

    # ------------------------------------------------------------
    # Macro AUC (AUTENTICAÇÃO) por identidade: média simples dos AUCs por classe
    # (equivalente ao "AUC composto" = mean(AUC one-vs-all) em autenticação)
    # ------------------------------------------------------------
    macro_auth_cfg = (auth_cfg.get("macro_auc", {}) or {})
    if bool(macro_auth_cfg.get("enable", False)):
        print("\n[AUTH] Macro ROC-AUC por identidade (one-vs-all, mesma identidade vs outras):")
        macro_res = auth_macro_auc_per_identity(
            emb=emb,
            P=P,
            y_true=y_eval,
            y_pred=y_pred,
            pmax=pmax,
            rng=rng,
            cfg=macro_auth_cfg,
        )
        if macro_res.get("n_classes_used", 0) > 0:
            print(
                f"  [AUTH] Macro ROC-AUC (média simples em {macro_res['n_classes_used']} classes): "
                f"cosine={macro_res['cos_mean']:.4f} (mediana={macro_res['cos_median']:.4f}) | "
                f"prob_dot={macro_res['prob_mean']:.4f} (mediana={macro_res['prob_median']:.4f}) | "
                f"id+conf={macro_res['conf_mean']:.4f} (mediana={macro_res['conf_median']:.4f})"
            )
        else:
            print("  [AUTH] Macro ROC-AUC: não há classes suficientes (precisa >=2 amostras por classe).")

# imprime resultados (métricas além de acurácia)
    cm_cos = {"TP": int(res_cos.get("TP", 0)), "TN": int(res_cos.get("TN", 0)), "FP": int(res_cos.get("FP", 0)),
              "FN": int(res_cos.get("FN", 0))}
    mets_cos = _binary_metrics_from_counts(cm_cos)
    print("\n[AUTH] Resultados (cosine-threshold):",
          {"thr": float(thr_cos), "mode": res_cos.get("mode"), "pairs": res_cos.get("pairs"), **mets_cos})
    print_confusion_binary("[AUTH] Confusão cosine (same vs different)", cm_cos)

    if "m_f1" in res_prob:
        print("[AUTH] Resultados (prob_dot-threshold):", {"thr": float(thr_prob), **{k: res_prob.get(k) for k in (
        "mode", "pairs", "m_acc", "m_f1", "m_balanced_acc", "TP", "TN", "FP", "FN")}})
        print_confusion_binary("[AUTH] Confusão prob_dot (same vs different)",
                               {"TP": int(res_prob["TP"]), "TN": int(res_prob["TN"]), "FP": int(res_prob["FP"]),
                                "FN": int(res_prob["FN"])})
    else:
        print("[AUTH] Resultados (prob_dot-threshold):", res_prob)

    if "m_f1" in res_conf:
        print("[AUTH] Resultados (id+conf):", {"thr_conf": float(thr_conf), **{k: res_conf.get(k) for k in (
        "mode", "pairs", "m_acc", "m_f1", "m_balanced_acc", "TP", "TN", "FP", "FN")}})
        print_confusion_binary("[AUTH] Confusão id+conf (same vs different)",
                               {"TP": int(res_conf["TP"]), "TN": int(res_conf["TN"]), "FP": int(res_conf["FP"]),
                                "FN": int(res_conf["FN"])})
    else:
        print("[AUTH] Resultados (id+conf):", res_conf)

    # matrizes por âncora (agora: cosine vs id+conf)
    n_anchor = int(auth_cfg.get("n_anchor_images", 10))
    anchor_idx = rng.choice(n, size=min(n_anchor, n), replace=False).astype(np.int64, copy=False)

    print("\n[AUTH] Matrizes por âncora (TP/FP/FN/TN vs todas as outras amostras):")
    for ai in anchor_idx.tolist():
        mask_other = np.ones(n, dtype=bool)
        mask_other[int(ai)] = False
        truth_same = (y_eval == y_eval[int(ai)]) & mask_other

        sims = emb @ emb[int(ai)]
        pred_same_cos = (sims >= float(thr_cos)) & mask_other
        cm_a_cos = _binary_counts_from_pred_truth(pred_same_cos, truth_same)
        mets_a_cos = _binary_metrics_from_counts(cm_a_cos)

        pred_same_conf = ((y_pred == y_pred[int(ai)]) & (
                    np.minimum(pmax, pmax[int(ai)]) >= float(thr_conf))) & mask_other
        cm_a_conf = _binary_counts_from_pred_truth(pred_same_conf, truth_same)
        mets_a_conf = _binary_metrics_from_counts(cm_a_conf)

        print(f"  âncora idx={int(ai)} | true_class={int(y_eval[int(ai)])}")
        print(f"    cosine: thr={float(thr_cos):.4f} | {mets_a_cos} | cm={cm_a_cos}")
        print(f"    id+conf: thr_conf={float(thr_conf):.4f} | {mets_a_conf} | cm={cm_a_conf}")

    # =========================
    # AUC macro (one-vs-rest) - todas as classes
    # =========================
    macro_auc_cfg = SCRIPT_CONFIG.get("macro_auc", {}) or {}
    if bool(macro_auc_cfg.get("enable", True)):
        classes_for_auc = np.unique(y_eval).astype(np.int64, copy=False)
        aucs = []
        use_fast = bool(macro_auc_cfg.get("use_fast_auc", True))
        prog_every = int(macro_auc_cfg.get("progress_every", 50))
        prog_every = max(0, prog_every)
        t0_auc = time.time()
        C = int(classes_for_auc.size)
        for ci, c in enumerate(classes_for_auc.tolist(), start=1):
            if prog_every > 0 and (ci == 1 or (ci % prog_every) == 0 or ci == C):
                elapsed = time.time() - t0_auc
                rate = (ci / elapsed) if elapsed > 0 else 0.0
                eta = ((C - ci) / rate) if rate > 0 else float('nan')
                print(f"[IDENTIFICAÇÃO] Macro AUC OVR: {ci}/{C} | elapsed={elapsed:.1f}s | ETA~{eta:.1f}s")

            idx_k = np.where(classes_modelo == int(c))[0]
            if idx_k.size == 0:
                continue
            k = int(idx_k[0])
            scores = P[:, k]
            truth = (y_eval == int(c)).astype(np.int8)

            if use_fast:
                auc = roc_auc_fast(scores, truth == 1)
                if np.isnan(auc):
                    continue
                aucs.append(float(auc))
            else:
                fpr, tpr, _thrs, auc = roc_curve_simple(scores, truth)
                if np.isnan(auc) or fpr.size == 0:
                    continue
                aucs.append(float(auc))

        if len(aucs) == 0:
            print("\n[IDENTIFICAÇÃO] Macro AUC (one-vs-rest): não foi possível calcular (sem curvas válidas).")
        else:
            macro_auc = float(np.mean(aucs))
            print("\n[IDENTIFICAÇÃO] Macro AUC (one-vs-rest) entre todas as classes (thresholds variados):")
            print(f"  macro_auc={macro_auc:.6f} | n_classes={len(aucs)} | min={min(aucs):.6f} | max={max(aucs):.6f}")

    # INTERATIVO
    if bool(SCRIPT_CONFIG["enable_interactive_queries"]):
        print("\n[INTERATIVO] Ligado. Digite 'q' para sair.")
        while True:
            cmd = input(
                "\nEscolha: (1) Predizer classe por ID | (2) Autenticar por dois IDs | (q) sair : ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break

            if cmd == "1":
                s = input("Digite o ID da amostra (0..N-1): ").strip()
                if s.lower() in ("q", "quit", "exit"):
                    break
                try:
                    idx = int(s)
                except Exception:
                    print("ID inválido.")
                    continue
                if idx < 0 or idx >= n:
                    print("Fora do intervalo.")
                    continue
                topk = np.argsort(-P[idx])[:5]
                print(f"  y_pred={int(y_pred[idx])} | pmax={float(P[idx].max()):.4f}")
                print("  top5:", [(int(classes_modelo[k]), float(P[idx][k])) for k in topk])

            elif cmd == "2":
                s1 = input("Digite o ID i (0..N-1): ").strip()
                if s1.lower() in ("q", "quit", "exit"):
                    break
                s2 = input("Digite o ID j (0..N-1): ").strip()
                if s2.lower() in ("q", "quit", "exit"):
                    break
                try:
                    i = int(s1);
                    j = int(s2)
                except Exception:
                    print("IDs inválidos.")
                    continue
                if i < 0 or i >= n or j < 0 or j >= n:
                    print("Fora do intervalo.")
                    continue

                sim_cos = float(np.dot(emb[i], emb[j]))
                sim_prob = float(np.dot(P[i], P[j]))
                same_cos = sim_cos >= float(thr_cos)
                same_prob = sim_prob >= float(thr_prob)
                same_conf = (int(y_pred[i]) == int(y_pred[j])) and (float(min(pmax[i], pmax[j])) >= float(thr_conf))

                print(f"  cosine_sim={sim_cos:.4f} | thr_cos={float(thr_cos):.4f} | same? {bool(same_cos)}")
                print(f"  prob_dot={sim_prob:.4f} | thr_prob={float(thr_prob):.4f} | same? {bool(same_prob)}")
                print(
                    f"  id+conf: yhat_i={int(y_pred[i])} (pmax={float(pmax[i]):.4f}) | yhat_j={int(y_pred[j])} (pmax={float(pmax[j]):.4f}) | thr_conf={float(thr_conf):.4f} | same? {bool(same_conf)}")
                if same_conf:
                    print(f"  ==> Prediz MESMA pessoa: classe={int(y_pred[i])}")
                else:
                    print("  ==> Prediz pessoas DIFERENTES")

            else:
                print("Comando inválido (use 1, 2 ou q).")

# ============================================================
# OUTPUT LOGGING (TEE)
# ============================================================
import sys
import io
import traceback
from datetime import datetime
from pathlib import Path

class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

def _run_with_output_log(main_func, log_prefix="cnn_ident_auth_eval"):
    out_dir = _resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = out_dir / f"{log_prefix}_output_{ts}.txt"

    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[RUN] {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"[SCRIPT] {Path(__file__).name if '__file__' in globals() else 'interactive'}\n")
            f.write("=" * 80 + "\n\n")
            f.flush()

            sys.stdout = _Tee(old_out, f)
            sys.stderr = _Tee(old_err, f)

            try:
                main_func()
            except SystemExit:
                raise
            except Exception:
                traceback.print_exc()
                raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err

    print(f"[LOG] Output salvo em: {log_path}")


if __name__ == "__main__":
     _run_with_output_log(main, log_prefix="cnn_ident_auth_eval")
