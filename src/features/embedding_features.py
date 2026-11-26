"""Dense text embedding extraction with length-sorted batching, caching, and similarity helpers.

Owned by Teammate C (T-023).

Public API
----------
embed(texts, model_name, batch, max_len, device, show_progress_bar, use_cache) -> np.ndarray
max_cosine_to_train(emb_query, emb_train, is_train) -> np.ndarray
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src import config

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = getattr(config, "CACHE_DIR", getattr(config, "OUTPUT_DIR", config.ROOT / "outputs") / "cache")


def _clean_model_tag(model_name: str) -> str:
    """Return a filesystem-safe tag for the model name."""
    clean = model_name.split("/")[-1]
    return clean.replace("-", "_").replace(".", "_")


def _compute_cache_key(texts: Sequence[str], model_name: str, max_len: int) -> str:
    """Compute deterministic MD5 hash for the combination of texts, model, and max_len."""
    hasher = hashlib.md5()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(max_len).encode("utf-8"))
    hasher.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        hasher.update(t.encode("utf-8", errors="replace"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def embed(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch: int = 64,
    max_len: int = 128,
    device: Optional[str] = None,
    show_progress_bar: bool = False,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Extract L2-normalised dense sentence embeddings with length-sorted batching and disk caching.

    Parameters
    ----------
    texts : Sequence[str]
        Input strings. Empty strings or None are safely mapped to "".
    model_name : str
        HuggingFace model repository or local path.
    batch : int
        Batch size for inference.
    max_len : int
        Maximum sequence length for tokenization.
    device : Optional[str]
        Device to run inference on ('cuda', 'mps', 'cpu', or auto-detected).
    show_progress_bar : bool
        Whether to display SentenceTransformer progress bar.
    use_cache : bool
        If True, reads/writes cached embeddings from disk.
    cache_dir : Optional[Path]
        Directory for cached `.npy` files.

    Returns
    -------
    np.ndarray
        Array of shape `(len(texts), embedding_dim)`, dtype float32, L2-normalised.
    """
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32)

    # Sanitize texts (handle None / NaN / whitespace)
    cleaned_texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in texts]

    # Prepend 'query: ' for e5 family models per specification
    is_e5 = "e5" in model_name.lower()
    encode_inputs = [f"query: {t}" if is_e5 else t for t in cleaned_texts]

    # Check disk cache
    target_cache_dir = cache_dir or CACHE_DIR
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    model_tag = _clean_model_tag(model_name)
    cache_key = _compute_cache_key(cleaned_texts, model_name, max_len)
    cache_file = target_cache_dir / f"emb_{model_tag}_{cache_key}.npy"

    if use_cache and cache_file.exists():
        log.info("Loading cached embeddings from %s", cache_file)
        return np.load(cache_file)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Encoding %d texts with %s on %s (batch=%d, max_len=%d)",
             len(encode_inputs), model_name, device, batch, max_len)

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len

    # Length-sorted batching for minimal padding overhead
    lengths = [len(t) for t in encode_inputs]
    sorted_idx = np.argsort(lengths)
    sorted_texts = [encode_inputs[i] for i in sorted_idx]

    # Use fp16 precision on CUDA for acceleration & memory efficiency
    use_fp16 = device == "cuda"
    if use_fp16:
        model.half()

    # Encode with dynamic padding
    encoded = model.encode(
        sorted_texts,
        batch_size=batch,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We will explicitly L2-normalise below
        device=device,
    )

    # Restore original order
    orig_order_idx = np.empty_like(sorted_idx)
    orig_order_idx[sorted_idx] = np.arange(len(sorted_idx))
    embeddings = encoded[orig_order_idx].astype(np.float32)

    # L2-normalisation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    embeddings = embeddings / norms

    # Save to cache
    if use_cache:
        log.info("Saving embeddings to cache %s", cache_file)
        np.save(cache_file, embeddings)

    return embeddings


def max_cosine_to_train(
    emb_query: np.ndarray,
    emb_train: np.ndarray,
    is_train: bool = False,
    chunk_size: int = 25_000,
) -> np.ndarray:
    """Compute maximum cosine similarity of query embeddings against training embeddings.

    For training rows (is_train=True or when query is train), cross-fitting is applied
    by masking the self-similarity diagonal so a row is never compared to itself.

    Parameters
    ----------
    emb_query : np.ndarray
        Query embeddings of shape (n_query, d), assumed L2-normalised.
    emb_train : np.ndarray
        Training embeddings of shape (n_train, d), assumed L2-normalised.
    is_train : bool
        If True, masks self-similarity (diagonal) with -2.0 so query rows compare
        only against other training rows.
    chunk_size : int
        Chunk size for query matrix multiplications to bound memory on large sets (e.g. 500k).

    Returns
    -------
    np.ndarray
        Array of shape (n_query,) containing the maximum cosine similarity to train.
    """
    n_query = len(emb_query)
    n_train = len(emb_train)
    if n_query == 0 or n_train == 0:
        return np.zeros(n_query, dtype=np.float32)

    # Detect if query and train are the exact same dataset
    if not is_train and (emb_query is emb_train or (n_query == n_train and np.array_equal(emb_query, emb_train))):
        is_train = True

    out = np.empty(n_query, dtype=np.float32)

    # Process in chunks to bound memory
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        q_chunk = emb_query[start:end]
        # Dot product of L2-normalised vectors equals cosine similarity
        sim_chunk = np.dot(q_chunk, emb_train.T)

        if is_train:
            # Mask diagonal in this chunk
            for row_idx in range(start, end):
                if row_idx < n_train:
                    sim_chunk[row_idx - start, row_idx] = -2.0

        out[start:end] = sim_chunk.max(axis=1)

    return out
