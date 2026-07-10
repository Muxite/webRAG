"""
Opt-in ConSol (SPRT) early-stopping pilot for leaf-extraction voting.

Background
----------
The compiled-graph benchmark decides a leaf's factual value by self-consistency voting:
``_vote_extract`` in ``execution_compiled.py`` fires a *fixed* ``k`` independent extractions
(``_votes_for_model``: cheap tier k=5, mid k=3, premium k=2) and takes the majority. The
campaign has repeatedly needed brute-force repeated live runs (R=3, sometimes R=5) to
separate signal from noise, which is slow and expensive on the cheap heavy-redundancy tier.

ConSol (https://github.com/LiuzLab/consol, ``pip install consol``) reframes "find the mode
of N sampled answers" as a Bernoulli hypothesis test between the two most-frequent responses
(SPRT / mixture-SPRT), stopping sampling as soon as the evidence crosses a decision boundary
instead of always spending a fixed N. Its confidence models are the reusable, LLM-agnostic
core: each exposes ``test(first, second) -> bool`` over the integer counts of the top-two
answers, plus ``config.max_trials``. We drive our OWN sampler with our OWN vote keys and only
borrow that statistical stopping rule — none of ConSol's langchain sampling machinery.

Discipline
----------
* Fully OPT-IN via ``IDEA_TEST_USE_CONSOL=1`` (default off). When unset, ``consol_vote``
  returns ``None`` immediately and the caller keeps its byte-identical fixed-k behavior.
* Never a hard dependency: if ``consol`` is not installed (or import errors), we log once and
  return ``None`` so the caller falls back to fixed-k. ``consol`` is intentionally NOT added
  to ``requirements.txt``.
* Conservative pilot: the sample cap defaults to ``k`` (overridable via
  ``IDEA_TEST_CONSOL_MAX_SAMPLES``), so an enabled run can only ever spend *fewer* samples
  than today's fixed-k baseline, never more. Early convergence saves samples; a
  never-converging leaf exhausts the cap rather than sampling forever.

This is a PILOT wrapper around a single call site (leaf extraction voting); it does not
replace ``_vote_extract``'s logic wholesale.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Awaitable, Callable, Optional

_logger = logging.getLogger(__name__)

ENABLE_ENV = "IDEA_TEST_USE_CONSOL"
MODEL_ENV = "IDEA_TEST_CONSOL_MODEL"          # msprt|sprt|pvalue|bayesian_posterior
MAX_SAMPLES_ENV = "IDEA_TEST_CONSOL_MAX_SAMPLES"

_CONFIDENCE_MODELS = {
    "msprt": "MsprtConfidenceModel",
    "sprt": "SprtConfidenceModel",
    "pvalue": "PValueConfidenceModel",
    "bayesian_posterior": "BayesianPosteriorConfidenceModel",
}

_import_warned = False


def consol_enabled() -> bool:
    """True only when the opt-in env flag is exactly ``1``. Default off."""
    return os.environ.get(ENABLE_ENV, "").strip() == "1"


def _load_confidence_model():
    """Lazily import a ConSol confidence model. Returns ``None`` (and warns once) if ConSol
    is not installed or errors — the caller then falls back to fixed-k. Selected by
    ``IDEA_TEST_CONSOL_MODEL`` (default ``msprt``)."""
    global _import_warned
    try:
        import consol.confidence_models as cm
    except Exception as exc:  # noqa: BLE001 - any import failure must fall back, not raise
        if not _import_warned:
            _logger.warning(
                "ConSol requested (%s=1) but unavailable (%s); falling back to fixed-k voting",
                ENABLE_ENV, exc,
            )
            _import_warned = True
        return None
    name = os.environ.get(MODEL_ENV, "msprt").strip().lower()
    cls_name = _CONFIDENCE_MODELS.get(name, _CONFIDENCE_MODELS["msprt"])
    try:
        return getattr(cm, cls_name)()
    except Exception as exc:  # noqa: BLE001
        if not _import_warned:
            _logger.warning("ConSol confidence model %s failed to init (%s); falling back", cls_name, exc)
            _import_warned = True
        return None


def _max_samples(k: int, max_trials: int) -> int:
    """Sample cap. Default ``k`` (never spends more than the fixed-k baseline), overridable
    with ``IDEA_TEST_CONSOL_MAX_SAMPLES``; always clamped to the model's ``max_trials``."""
    override = os.environ.get(MAX_SAMPLES_ENV, "").strip()
    cap = max(1, int(override)) if override.isdigit() else max(1, k)
    return min(cap, max_trials)


def _default_is_unknown(ans: str) -> bool:
    return (not ans) or ans.strip().upper().startswith("UNKNOWN")


class ConsolVoteResult:
    """Outcome of a ConSol-driven vote. ``answer`` is the chosen value ('' if every sample was
    UNKNOWN), ``num_samples`` the actual number of samples drawn (<= the fixed-k cap), and
    ``confident`` whether the SPRT test triggered an early stop (vs exhausting the cap)."""
    __slots__ = ("answer", "num_samples", "confident")

    def __init__(self, answer: str, num_samples: int, confident: bool):
        self.answer = answer
        self.num_samples = num_samples
        self.confident = confident


async def consol_vote(
    sample: Callable[[float], Awaitable[str]],
    *,
    k: int,
    key_fn: Callable[[str], str],
    is_unknown_fn: Callable[[str], bool] = _default_is_unknown,
    anchor_temp: float = 0.0,
    diverse_temp: float = 0.3,
    confidence_model=None,
) -> Optional[ConsolVoteResult]:
    """Sequentially draw samples and stop as soon as ConSol's SPRT test is confident in the
    leading answer, aggregating exactly like ``_vote_extract`` (anchor-at-temp-0, ties break to
    the anchor). Sampling is sequential (not the fixed-k parallel gather) because early stopping
    needs to inspect the running tally after each draw.

    Returns ``None`` — meaning "caller keeps its fixed-k behavior" — when the opt-in flag is off
    or ConSol is unavailable. Otherwise returns a :class:`ConsolVoteResult`.

    ``sample(temperature)`` is an async callback yielding one raw answer string; the first draw
    uses ``anchor_temp``, the rest ``diverse_temp`` (mirroring ``_vote_extract``).
    """
    if not consol_enabled():
        return None
    cm = confidence_model if confidence_model is not None else _load_confidence_model()
    if cm is None:
        return None

    cap = _max_samples(k, cm.config.max_trials)
    answers: list[str] = []
    counts: Counter[str] = Counter()
    confident = False
    n = 0
    while n < cap:
        temp = anchor_temp if n == 0 else diverse_temp
        try:
            raw = await sample(temp)
        except Exception as exc:  # noqa: BLE001 - a flaky sample shouldn't abort the vote
            _logger.warning("ConSol sample %d/%d errored (%s); continuing", n + 1, cap, exc)
            n += 1
            continue
        n += 1
        answers.append(raw)
        if is_unknown_fn(raw):
            continue
        counts[key_fn(raw)] += 1
        top = counts.most_common(2)
        first = top[0][1] if top else 0
        second = top[1][1] if len(top) > 1 else 0
        if first > 0 and cm.test(first, second):
            confident = True
            break

    cands = [a for a in answers if not is_unknown_fn(a)]
    if not cands:
        return ConsolVoteResult("", n, confident)
    top_count = counts.most_common(1)[0][1]
    tied = {key for key, c in counts.items() if c == top_count}
    anchor = answers[0] if (answers and not is_unknown_fn(answers[0])) else ""
    chosen_key = key_fn(anchor) if anchor and key_fn(anchor) in tied else counts.most_common(1)[0][0]
    chosen = next(a for a in cands if key_fn(a) == chosen_key)
    return ConsolVoteResult(chosen, n, confident)
