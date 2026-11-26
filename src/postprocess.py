"""Consistency post-processing rules between Task A, B, C outputs.

Owned by Teammate B (T-024, Phase 2).

Rules are derived from the EDA cross-tab (label × techniques × action).
Only rules that hold in ≥95% of real labeled rows are enforced as hard rules.
Hypotheses (not yet verified) become soft priors.
Rules must NEVER key off specific text strings or test ids.
"""


def apply_rules(label: int, tech, action: int, probs) -> tuple:
    """Apply consistency rules learned from data cross-tab.

    label:  predicted Task A label (0–5)
    tech:   (6,) binary technique array
    action: predicted Task C action (0–5)
    probs:  dict or namedtuple with 'A', 'B', 'C' probability arrays

    Returns: (label, tech, action) possibly corrected.
    Raises NotImplementedError until T-024 is complete.
    """
    raise NotImplementedError("postprocess.apply_rules — implement in T-024 after EDA cross-tab")
