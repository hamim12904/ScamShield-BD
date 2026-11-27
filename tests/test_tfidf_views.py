"""Tests for the three Model 1 text views produced by build_text_dict."""
import pandas as pd

from src.features.tfidf_features import build_text_dict


def test_build_text_dict_uses_distinct_norm_and_skeleton_views():
    text = "Please v.e.r.i.f.y OTP 482913 now"
    views = build_text_dict(pd.DataFrame({"id": [1], "text": [text]}))

    assert "verify" in views["text_skel"][0]
    assert "<N6>" in views["text_skel"][0]
    assert "482913" in views["text_norm"][0]
    assert views["text_norm"] != views["text_skel"]
    assert all(len(value) == 1 for value in views.values())
