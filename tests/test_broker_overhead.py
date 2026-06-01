"""Sidecar-overhead microbenchmark sanity gate (Task 6 / E4 panel (b)).

We do not assert a wall-time budget here — those numbers are
environment-dependent (CI runner CPU, contention, GC pressure). What we
*can* gate is that the augmenter's cost grows roughly linearly with
input size and not catastrophically (e.g., quadratically) — that is the
sidecar-overhead claim that backs Figure 1 panel (b).

Approach: time augmentation at two padded input sizes (small + medium)
and assert the medium/small ratio stays under a generous ceiling. A
genuine O(n²) regression in the augmenter's clone or walk would blow
through that ceiling immediately. Set the ceiling high enough that
normal CI jitter is not a flake source.

Skipped (not failed) when the rich_full synthetic input is missing
(i.e., `python -m eval.build_corpus` has not run).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RICH_INPUT = REPO_ROOT / "eval" / "corpus" / "synthetic" / "rich_full.jsonld"


def _missing_corpus_reason() -> str | None:
    if not RICH_INPUT.exists():
        return (
            f"corpus base {RICH_INPUT} missing; "
            f"run `python -m eval.build_corpus` first"
        )
    return None


_SKIP_REASON = _missing_corpus_reason()


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_augmentation_cost_scales_subquadratically() -> None:
    """Augmenter wall time at 4x input padding must stay under 12x the
    baseline. A 12x ceiling at 4x size lets linear (≈4x) and modestly
    super-linear behaviour pass while catching a true O(n²) regression
    (≈16x). Wide enough to absorb GC jitter on slow CI runners.
    """
    from eval.run_e4_perf import _time_one
    import json

    base = json.loads(RICH_INPUT.read_text(encoding="utf-8"))
    small = _time_one(base, n_pad=25, repeats=10)
    big = _time_one(base, n_pad=100, repeats=10)

    # Use p95 to avoid a single-fast-baseline-run causing a flake when
    # the medium run hits an unrelated GC pause.
    ratio = big.p95_ms / max(small.p95_ms, 1e-3)
    assert ratio < 12.0, (
        f"augmentation cost grew {ratio:.1f}x at 4x input size — "
        f"super-linear regression? small p95={small.p95_ms:.2f}ms "
        f"big p95={big.p95_ms:.2f}ms"
    )


@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
def test_padded_input_round_trips_through_augmenter() -> None:
    """The padding the perf runner injects must not break the augmenter
    — otherwise panel (b)'s timings are measuring an error path. Single
    augment call at the largest pad size; success is the assertion.
    """
    import json

    from eval.run_e4_perf import _padded_input
    from src.metadata.augmenter import augment_ai_ready
    from src.metadata.types import SegmentationType

    base = json.loads(RICH_INPUT.read_text(encoding="utf-8"))
    body = _padded_input(base, n_predicates=400)
    augmented = augment_ai_ready(
        body,
        job_id="perf-padded-sanity",
        segmentation_type=SegmentationType.INSTANCE,
        coco_access_url="https://example.org/coco/perf-padded-sanity.json",
    )
    assert isinstance(augmented, dict)
    dataset = augmented["@graph"][0]
    # The 400 pad predicates must ride through (zero-loss applies to
    # padded predicates too — they're just custom-namespace leaves).
    pad_keys = [k for k in dataset if isinstance(k, str) and k.startswith("evpad:pad")]
    assert len(pad_keys) == 400, f"expected 400 pad predicates, got {len(pad_keys)}"
