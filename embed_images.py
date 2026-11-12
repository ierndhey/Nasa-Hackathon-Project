"""Extract resumable, CPU DINOv2 Base embeddings from local satellite tiles."""

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

# Use standard HTTPS downloads; the optional Xet client can stall on this network.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from filelock import FileLock, Timeout
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "facebook/dinov2-base"
# Meta's Base checkpoint, pinned instead of following updates to 'main'.
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
EMBEDDING_DIM = 768
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
LOGGER = logging.getLogger(__name__)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_DIR / "resources" / "zoom_8")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_DIR / "resources" / "embeddings" / "dinov2_base_zoom_8")
    parser.add_argument("--cache-dir", type=Path,
                        default=PROJECT_DIR / "resources" / "model_cache" / "huggingface")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int,
                        default=min(4, os.cpu_count() or 1), help="CPU inference threads (default: up to 4)")
    parser.add_argument("--limit", type=positive_int,
                        help="Attempt at most this many new/changed images; cached images do not count")
    return parser.parse_args()


def make_transform():
    # Matches DINOv2's classification evaluation transform. No random augmentation.
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def extraction_config(input_dir):
    return {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "loader": "transformers.AutoModel, safetensors, native PyTorch SDPA",
        "embedding": "CLS token, L2 normalized",
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float32",
        "device": "cpu",
        "preprocessing": {
            "orientation": "EXIF transpose", "color": "RGB",
            "resize_shorter_edge": 256, "interpolation": "bicubic",
            "center_crop": [224, 224],
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        },
        "packages": {name: version(name) for name in ("torch", "torchvision", "transformers", "numpy", "Pillow")},
    }


def prepare_database(connection, config):
    connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("""CREATE TABLE IF NOT EXISTS images (
        path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
        embedding BLOB, error TEXT
    )""")
    serialized = json.dumps(config, sort_keys=True)
    previous = connection.execute("SELECT value FROM settings WHERE key='config'").fetchone()
    if previous and previous[0] != serialized:
        if connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]:
            raise RuntimeError("Extraction configuration differs from this cache. Use a new --output-dir.")
    connection.execute("INSERT OR REPLACE INTO settings VALUES ('config', ?)", (serialized,))
    connection.execute("""CREATE TEMP TABLE current_files (
        path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL
    )""")
    connection.commit()


def scan_images(input_dir):
    inventory = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                stat = path.stat()
                inventory.append((path.name, stat.st_size, stat.st_mtime_ns))
            except OSError as error:
                LOGGER.warning("Cannot inspect %s: %s", path, error)
    return inventory


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def save_failure(connection, record, error):
    # Clear any older embedding when a changed image can no longer be decoded.
    with connection:
        connection.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?, NULL, ?)",
                           (*record, str(error)))
    LOGGER.warning("Skipping %s: %s", record[0], error)


def encode_batch(model, tensors):
    with torch.inference_mode():
        features = model(torch.stack(tensors)).float()
        if features.shape != (len(tensors), EMBEDDING_DIM):
            raise RuntimeError(f"Unexpected embedding shape: {tuple(features.shape)}")
        norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
        if not torch.isfinite(features).all() or not (norms > 0).all():
            raise RuntimeError("Model produced invalid embeddings")
        return (features / norms).cpu().numpy()


def load_model(cache_dir):
    # Built-in Transformers architecture; no downloaded Python model code executes.
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, cache_dir=str(cache_dir),
        use_safetensors=True, trust_remote_code=False, attn_implementation="sdpa",
    ).to(device="cpu", dtype=torch.float32).eval()
    LOGGER.info("Loaded %s, revision %s", MODEL_NAME, MODEL_REVISION)

    def cls_embedding(pixel_values):
        return model(pixel_values=pixel_values).last_hidden_state[:, 0, :]

    return cls_embedding


def process_images(connection, pending, args):
    if not pending:
        LOGGER.info("All current images are cached; no model loading needed.")
        return 0, False
    torch.set_num_threads(args.threads)
    LOGGER.info("Loading pretrained DINOv2 Base on CPU; first run downloads the model.")
    model = load_model(args.cache_dir)
    transform = make_transform()
    processed = 0
    started = time.perf_counter()
    interrupted = False
    try:
        with tqdm(total=len(pending), desc="Embedding", unit="image") as progress:
            for start in range(0, len(pending), args.batch_size):
                batch_records, tensors = [], []
                for record in pending[start:start + args.batch_size]:
                    path = args.input_dir / record[0]
                    try:
                        if signature(path) != record[1:]:
                            raise OSError("Image changed since scanning; retry on the next run")
                        with Image.open(path) as image:
                            tensor = transform(ImageOps.exif_transpose(image).convert("RGB"))
                        if signature(path) != record[1:]:
                            raise OSError("Image changed while reading; retry on the next run")
                    except (OSError, Image.DecompressionBombError) as error:
                        save_failure(connection, record, error)
                        progress.update(1)
                        continue
                    batch_records.append(record)
                    tensors.append(tensor)
                if not tensors:
                    continue
                vectors = encode_batch(model, tensors)
                with connection:
                    for record, vector in zip(batch_records, vectors):
                        try:
                            if signature(args.input_dir / record[0]) != record[1:]:
                                raise OSError("Image changed during inference; retry on the next run")
                        except OSError as error:
                            save_failure(connection, record, error)
                            continue
                        connection.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, NULL)",
                                           (*record, vector.astype("<f4").tobytes()))
                        processed += 1
                progress.update(len(tensors))
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Interrupted. Committed embeddings are preserved; rerun to resume.")
    elapsed = time.perf_counter() - started
    LOGGER.info("Saved %d new embeddings in %.1f seconds (model loading excluded).", processed, elapsed)
    return processed, interrupted


CURRENT_ROWS = """FROM images AS i JOIN current_files AS c ON i.path=c.path
    AND i.size_bytes=c.size_bytes AND i.mtime_ns=c.mtime_ns"""


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_results(connection, output_dir, config, inventory_size):
    count = connection.execute(f"SELECT COUNT(*) {CURRENT_ROWS} WHERE i.embedding IS NOT NULL").fetchone()[0]
    array_temp = output_dir / "embeddings.npy.tmp"
    index_temp = output_dir / "images.csv.tmp"
    errors_temp = output_dir / "errors.csv.tmp"
    metadata_temp = output_dir / "metadata.json.tmp"
    if count:
        matrix = np.lib.format.open_memmap(array_temp, mode="w+", dtype=np.float32,
                                          shape=(count, EMBEDDING_DIM))
    else:
        with array_temp.open("wb") as target:
            np.save(target, np.empty((0, EMBEDDING_DIM), dtype=np.float32))
        matrix = None
    try:
        with index_temp.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["embedding_row", "relative_path", "size_bytes", "mtime_ns", "zoom", "tile_row", "tile_col"])
            rows = connection.execute(f"""SELECT i.path, i.size_bytes, i.mtime_ns, i.embedding
                {CURRENT_ROWS} WHERE i.embedding IS NOT NULL ORDER BY i.path""")
            for index, (path, size, mtime, blob) in enumerate(rows):
                vector = np.frombuffer(blob, dtype="<f4")
                if vector.shape != (EMBEDDING_DIM,):
                    raise RuntimeError(f"Invalid cached embedding for {path}")
                matrix[index] = vector
                tile = re.fullmatch(r"moon_z(\d+)_r(\d+)_c(\d+)\.[^.]+", path)
                writer.writerow([index, path, size, mtime, *(tile.groups() if tile else ("", "", ""))])
    finally:
        if matrix is not None:
            matrix.flush()
            del matrix  # Close the memory map before replacing its file on Windows.
    with errors_temp.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["relative_path", "error"])
        writer.writerows(connection.execute(f"SELECT i.path, i.error {CURRENT_ROWS} WHERE i.error IS NOT NULL ORDER BY i.path"))
    artifacts = {}
    for temporary, name in ((array_temp, "embeddings.npy"), (index_temp, "images.csv"), (errors_temp, "errors.csv")):
        artifacts[name] = sha256_file(temporary)
        temporary.replace(output_dir / name)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "shape": [count, EMBEDDING_DIM],
        "images_in_snapshot": inventory_size,
        "artifact_sha256": artifacts,
    }
    metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_temp.replace(output_dir / "metadata.json")
    LOGGER.info("Exported %d vectors to %s", count, output_dir)
    return count


def run(args):
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.input_dir}")
    inventory = scan_images(args.input_dir)
    if not inventory:
        LOGGER.warning("No supported images in %s; exporting an empty snapshot.", args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # A real OS lock releases automatically if the process exits or is killed.
    with FileLock(args.output_dir / "extraction.lock", timeout=0):
        connection = sqlite3.connect(args.output_dir / "checkpoint.sqlite3")
        try:
            config = extraction_config(args.input_dir)
            prepare_database(connection, config)
            with connection:
                connection.executemany("INSERT INTO current_files VALUES (?, ?, ?)", inventory)
            cached = {row[0]: row[1:] for row in connection.execute(
                "SELECT path, size_bytes, mtime_ns FROM images WHERE embedding IS NOT NULL")}
            pending = [record for record in inventory if cached.get(record[0]) != record[1:]]
            LOGGER.info("Found %d images; %d cached; %d need embeddings.",
                        len(inventory), len(inventory) - len(pending), len(pending))
            if args.limit is not None:
                pending = pending[:args.limit]
            LOGGER.info("CPU threads: %d; batch size: %d; this run: %d images.",
                        args.threads, args.batch_size, len(pending))
            _, interrupted = process_images(connection, pending, args)
            # Do not publish cached vectors for files deleted/replaced during a long run.
            inventory = scan_images(args.input_dir)
            with connection:
                connection.execute("DELETE FROM current_files")
                connection.executemany("INSERT INTO current_files VALUES (?, ?, ?)", inventory)
            export_results(connection, args.output_dir, config, len(inventory))
            failures = connection.execute(f"SELECT COUNT(*) {CURRENT_ROWS} WHERE i.error IS NOT NULL").fetchone()[0]
            if failures:
                LOGGER.warning("%d images failed; see errors.csv. They will be retried on the next run.", failures)
            return 130 if interrupted else (1 if failures else 0)
        finally:
            connection.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    args = parse_args()
    try:
        return run(args)
    except Timeout:
        LOGGER.error("Another embedding process is using this output directory.")
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; existing checkpoints are preserved.")
        return 130
    except Exception:
        LOGGER.exception("Embedding extraction stopped")
        return 1


if __name__ == "__main__":
    sys.exit(main())



