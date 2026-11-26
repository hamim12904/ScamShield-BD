import numpy as np
import pytest

from src.features.embedding_features import embed, max_cosine_to_train


def test_embed_basic():
    texts = ["hello world", "", "short", "a much longer message discussing financial mobile services in Bangladesh"]
    embs = embed(texts, batch=2, max_len=32, use_cache=False)
    assert embs.shape[0] == len(texts)
    assert embs.shape[1] > 0
    # Check L2-normalisation
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_max_cosine_to_train():
    # 3 dummy unit vectors in 2D
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    v3 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)], dtype=np.float32)

    train = np.stack([v1, v2, v3])

    # When query is train, self-comparison should be masked
    sims = max_cosine_to_train(train, train, is_train=True)
    assert len(sims) == 3
    # v1 against {v2, v3}: max is dot(v1, v3) = cos(pi/4) approx 0.7071
    np.testing.assert_allclose(sims[0], np.cos(np.pi / 4), atol=1e-5)
    # v2 against {v1, v3}: max is dot(v2, v3) = sin(pi/4) approx 0.7071
    np.testing.assert_allclose(sims[1], np.sin(np.pi / 4), atol=1e-5)

    # When query is test vector identical to v1
    test_q = np.array([[1.0, 0.0]], dtype=np.float32)
    test_sim = max_cosine_to_train(test_q, train, is_train=False)
    np.testing.assert_allclose(test_sim[0], 1.0, atol=1e-5)
