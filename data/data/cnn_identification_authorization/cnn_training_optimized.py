# =========================
# CONFIG (add these keys)
# =========================
CONFIG.update({
    # Perf knobs
    "MIXED_PRECISION": True,          # set False if using a GPU backend that doesn't like fp16
    "XLA": True,                      # jit_compile for train/eval steps
    "ALLOW_TF32": True,               # Ampere+ NVIDIA: enables TF32 matmul speedups
    "DETERMINISTIC": False,           # tf.data determinism (False = faster)

    # Input / resizing behavior
    "ASSUME_INGESTED_SIZE": True,     # if your ingestion already produced IMG_SIZE×IMG_SIZE, skip resize
    "STRICT_RESOLUTION_CHECK": False, # True => raise if any sampled image isn't IMG_SIZE×IMG_SIZE
    "RESOLUTION_SAMPLE_N": 256,       # how many images to sample for resolution confirmation

    # Debug / counters
    "LOG_EVERY_N_STEPS": 50,          # prints progress every N steps
    "PRINT_FIRST_BATCH_INFO": True,   # prints dtype/shape/device once per fold
})

# =========================
# PERFORMANCE SETUP
# =========================
def setup_tf_performance(cfg):
    # TF32 can be a free speed boost on Ampere+ GPUs (no visible quality drop in most CNNs)
    if cfg.get("ALLOW_TF32", False):
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
        except Exception:
            pass

    # Mixed precision
    if cfg.get("MIXED_PRECISION", False):
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled: mixed_float16")
        except Exception as e:
            print("[WARN] Could not enable mixed precision:", e)

# =========================
# RESOLUTION CONFIRMATION
# =========================
def confirm_image_resolution(df: pd.DataFrame, cfg, sample_n: int | None = None) -> None:
    """
    Samples images from manifest and prints distribution of decoded (H,W).
    This confirms *ingested* resolution (before any resize).
    """
    n = int(sample_n or cfg.get("RESOLUTION_SAMPLE_N", 256))
    n = min(n, len(df))
    if n <= 0:
        print("[WARN] Resolution check skipped (empty df).")
        return

    sample = df.sample(n=n, random_state=int(cfg["SEED"]))["dst"].tolist()

    counts = {}
    bad = []
    target = (int(cfg["IMG_SIZE"]), int(cfg["IMG_SIZE"]))

    for p in sample:
        raw = tf.io.read_file(p)
        img = tf.image.decode_jpeg(raw, channels=int(cfg["IN_CHANNELS"]))
        h = int(img.shape[0] or tf.shape(img)[0].numpy())
        w = int(img.shape[1] or tf.shape(img)[1].numpy())
        key = (h, w)
        counts[key] = counts.get(key, 0) + 1
        if key != target:
            bad.append((p, key))

    # Print summary
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("[INFO] Ingested resolution check (sample):")
    for (h, w), c in top:
        print(f"  - {h}x{w}: {c}/{n}")
    if len(counts) > 10:
        print(f"  ... ({len(counts)} unique resolutions in sample)")

    if bad:
        print(f"[WARN] Found {len(bad)}/{n} images not equal to target {target}. Examples:")
        for p, (h, w) in bad[:5]:
            print(f"  - {p} -> {h}x{w}")

        if cfg.get("STRICT_RESOLUTION_CHECK", False):
            raise ValueError(
                f"STRICT_RESOLUTION_CHECK enabled: found images not {target}. "
                "Set STRICT_RESOLUTION_CHECK=False or fix ingestion."
            )
    else:
        print(f"[INFO] All sampled images match target resolution {target} ✅")

# =========================
# TF.DATA PIPELINE (FASTER)
# =========================
def make_tf_dataset(df: pd.DataFrame, cfg: Dict[str, Any], training: bool):
    paths = df["dst"].to_numpy().astype(str)
    labels = df["y"].to_numpy().astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 20000),
            seed=int(cfg["SEED"]),
            reshuffle_each_iteration=True
        )

    mean = tf.constant(cfg["NORM_MEAN"], dtype=tf.float32)[None, None, :]
    std  = tf.constant(cfg["NORM_STD"],  dtype=tf.float32)[None, None, :]

    img_size = int(cfg["IMG_SIZE"])
    channels = int(cfg["IN_CHANNELS"])
    assume_ingested = bool(cfg.get("ASSUME_INGESTED_SIZE", False))

    @tf.function
    def _load(path, y):
        raw = tf.io.read_file(path)

        # Faster JPEG decode option (often helps CPU-side decode)
        img = tf.image.decode_jpeg(raw, channels=channels, dct_method="INTEGER_FAST")

        # If you trust ingestion is already correct size, skip resize.
        # Still set static shape to help XLA & kernel selection.
        if assume_ingested:
            img = tf.ensure_shape(img, [img_size, img_size, channels])
        else:
            img = tf.image.resize(img, [img_size, img_size], method="bilinear")
            img = tf.cast(img, tf.uint8)

        # Augmentation (do on uint8 then convert, or do on float32; both OK)
        if training:
            if cfg.get("AUG_HFLIP", True):
                img = tf.image.random_flip_left_right(img)
            pad = int(cfg.get("AUG_PAD", 0))
            if pad > 0:
                img = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
                img = tf.image.random_crop(img, size=[img_size, img_size, channels])

        # Convert to float32 [0,1] and normalize
        img = tf.image.convert_image_dtype(img, tf.float32)  # /255.0
        img = (img - mean) / std
        return img, y

    # Map in parallel
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    bs = int(cfg["BATCH_SIZE"])
    ds = ds.batch(bs, drop_remainder=training)

    # Options for speed
    options = tf.data.Options()
    options.experimental_deterministic = bool(cfg.get("DETERMINISTIC", False))
    ds = ds.with_options(options)

    # Prefetch
    if bool(cfg.get("PREFETCH", True)):
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# =========================
# FAST TRAIN/EVAL WITH COUNTERS
# =========================
def _make_optimizer(cfg):
    base = tf.keras.optimizers.Adam(learning_rate=float(cfg["LR"]))
    if cfg.get("MIXED_PRECISION", False):
        from tensorflow.keras import mixed_precision
        return mixed_precision.LossScaleOptimizer(base)
    return base

def train_one_epoch(model: Model, ds, optimizer, loss_fn, cfg: Dict[str, Any], fold_i: int, epoch: int):
    log_every = int(cfg.get("LOG_EVERY_N_STEPS", 50))
    xla = bool(cfg.get("XLA", False))

    # Compiled train step
    @tf.function(jit_compile=xla)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
            if model.losses:
                loss += tf.add_n(model.losses)

        if hasattr(optimizer, "get_scaled_loss"):
            scaled_loss = optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(y, tf.int32)), tf.int32))
        batch_n = tf.shape(y)[0]
        return loss, correct, batch_n, logits

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    t_epoch0 = time.perf_counter()
    t_last = t_epoch0

    first_batch_printed = False

    for step, (x, y) in enumerate(ds, start=1):
        loss, correct, batch_n, logits = train_step(x, y)

        # NOTE: loss is per-batch mean; we weight by batch size for epoch mean
        bn = int(batch_n.numpy())
        total_loss += float(loss.numpy()) * bn
        total_correct += int(correct.numpy())
        total_seen += bn

        if cfg.get("PRINT_FIRST_BATCH_INFO", True) and not first_batch_printed:
            first_batch_printed = True
            try:
                print(f"[DBG] Fold {fold_i} Epoch {epoch} first batch:")
                print(f"      x.shape={x.shape} x.dtype={x.dtype} x.device={x.device}")
                print(f"      y.shape={y.shape} y.dtype={y.dtype}")
            except Exception:
                pass

        if log_every > 0 and (step % log_every == 0):
            now = time.perf_counter()
            dt = now - t_last
            img_s = (log_every * int(cfg["BATCH_SIZE"])) / max(dt, 1e-9)
            acc = total_correct / max(total_seen, 1)
            print(
                f"[Fold {fold_i}][Epoch {epoch}] step {step} | "
                f"seen={total_seen} | acc={acc:.4f} | "
                f"{img_s:.1f} img/s | {dt:.2f}s/{log_every} steps"
            )
            t_last = now

    epoch_seconds = time.perf_counter() - t_epoch0
    mean_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    err = 1.0 - acc
    return mean_loss, acc, err, epoch_seconds

def evaluate(model: Model, ds, loss_fn, cfg: Dict[str, Any]):
    xla = bool(cfg.get("XLA", False))

    @tf.function(jit_compile=xla)
    def val_step(x, y):
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        if model.losses:
            loss += tf.add_n(model.losses)
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

# =========================
# IN run_kfold_cv(), add:
# =========================
# right after set_seed(...) and df loaded:
#   setup_tf_performance(cfg)
#   confirm_image_resolution(df, cfg)
#
# and replace optimizer/loss usage:
#   optimizer = _make_optimizer(cfg)
#   loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
# and replace epoch loop calls:
#   tr_loss, tr_acc, tr_err, dt = train_one_epoch(...)
#   va_loss, va_acc, va_err = evaluate(...)
