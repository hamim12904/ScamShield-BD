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


def test_cross_validate_embeddings_with_extra_and_leak_filter():
    import pandas as pd
    from src.training.train_embeddings import cross_validate_embeddings

    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "label": [0, 1, 0, 1],
        "techniques": ["NONE", "T1|T2", "NONE", "T2"],
        "action": [0, 1, 0, 1],
    })
    folds = pd.DataFrame({"fold_r0": [0, 0, 1, 1]})
    emb_train = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
        [0.1, 0.9],
    ], dtype=np.float32)
    extra_df = pd.DataFrame({
        "id": [901, 902],
        "label": [0, 1],
        "techniques": ["NONE", "T1"],
        "action": [0, 1],
        "weight": [0.3, 0.3],
        "near_real_ids": [[1], []],
    })
    emb_extra = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    p_a, p_b, p_c = cross_validate_embeddings(
        df, emb_train, folds, C=1.0, extra_df=extra_df, emb_extra=emb_extra
    )
    assert p_a.shape == (4, 6)
    assert p_b.shape == (4, 6)
    assert p_c.shape == (4, 6)
    assert np.allclose(p_a.sum(axis=1), 1.0)
