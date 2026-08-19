import dataclasses
import json
import re
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse


@dataclasses.dataclass(frozen=True)
class Verdict:
    correct: bool
    parsed: Optional[str]
    parse_failed: bool
    abstained: bool


def normalize_url(url: str) -> str:
    """Normalize URL by ignoring scheme, case, trailing slash, fragment, and query string."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    return urlunparse(("", netloc, path, "", "", ""))


def grade_url(raw: str, expected_url: str) -> Verdict:
    if not raw or not raw.strip():
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    url_pattern = r'https?://[^\s<>"\')]+'
    matches = re.findall(url_pattern, raw)
    if not matches:
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    found_url = matches[-1].rstrip(".,;!?:")
    norm_found = normalize_url(found_url)
    norm_expected = normalize_url(expected_url)

    correct = (norm_found == norm_expected)
    return Verdict(correct=correct, parsed=found_url, parse_failed=False, abstained=False)


def grade_regex(raw: str, pattern: str) -> Verdict:
    if not raw or not raw.strip():
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    match = re.search(pattern, raw)
    if match:
        return Verdict(correct=True, parsed=match.group(0), parse_failed=False, abstained=False)
    else:
        return Verdict(correct=False, parsed=raw.strip(), parse_failed=False, abstained=False)


def _json_candidates(raw: str) -> List[str]:
    """Substrings of a completion that might parse as a JSON object.

    Fenced blocks first, then bare braces, so a model that both explains itself and
    emits a code fence is read from the fence.
    """
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = list(fenced)
    candidates.extend(re.findall(r"\{[^{}]*\}", raw, re.DOTALL))
    return candidates


def _extract_from_json(raw: str) -> Optional[str]:
    for cand in _json_candidates(raw):
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                # Look for common verdict keys
                for k in ["verdict", "answer", "choice", "prediction", "label", "result"]:
                    if k in data and isinstance(data[k], str):
                        return data[k]
                # If single value in dict
                for v in data.values():
                    if isinstance(v, str):
                        return v
        except Exception:
            continue
    return None


def extract_confidence(raw: str) -> Optional[float]:
    """Pull a stated confidence out of a completion, or None if there isn't one.

    Two shapes are accepted, because the arms ask for two: a numeric
    ``"confidence"`` field, and ``C_verbal``'s ``"certainty"`` band. The band
    mapping lives in ``calibration.VERBAL_BANDS`` and is declared in source --
    inferring it from the observed outcomes would let the arm be tuned after the
    fact into whatever ordering best fit the data.

    Out-of-range numbers are rejected rather than clamped. A model answering "95"
    for 95% is not stating a probability, and clamping it to 1.0 would silently
    convert a formatting failure into maximal confidence -- the single most
    damaging direction for a calibration metric to be wrong in.
    """
    if not raw or not raw.strip():
        return None
    from agent.app.promptbench.calibration import VERBAL_BANDS, is_finite_probability

    for cand in _json_candidates(raw):
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "confidence" in data and is_finite_probability(data["confidence"]):
            return float(data["confidence"])
        band = data.get("certainty")
        if isinstance(band, str) and band.strip().lower() in VERBAL_BANDS:
            return VERBAL_BANDS[band.strip().lower()]

    m = re.search(r'"?confidence"?\s*[:=]\s*(0?\.\d+|0|1(?:\.0+)?)', raw, re.IGNORECASE)
    if m:
        value = float(m.group(1))
        if is_finite_probability(value):
            return value

    m = re.search(r"\b(certain|likely|unsure|guessing)\b", raw, re.IGNORECASE)
    if m:
        return VERBAL_BANDS[m.group(1).lower()]
    return None


def grade_confidence(
    raw: str,
    expected: str,
    choices: Sequence[str],
    abstain_choices: Optional[Sequence[str]] = None,
) -> Tuple[Verdict, Optional[float]]:
    """Grade a calibration arm: the answer, plus the confidence it stated.

    The answer goes through ``grade_enum`` unchanged, so a calibration cell's
    accuracy is measured exactly like an A-arm cell's and the two remain
    comparable. A missing confidence is ``None`` -- reported as its own column,
    never defaulted to 0.5, which would fabricate a perfectly average judge out
    of a model that said nothing.
    """
    return grade_enum(raw, expected, choices, abstain_choices), extract_confidence(raw)


def grade_enum(
    raw: str,
    expected: str,
    choices: Sequence[str],
    abstain_choices: Optional[Sequence[str]] = None,
) -> Verdict:
    if not raw or not raw.strip():
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    abstain_set = set(abstain_choices) if abstain_choices else set()
    choice_map = {c.lower(): c for c in choices}

    # 1. Check for JSON structure
    json_val = _extract_from_json(raw)
    if json_val is not None and json_val.lower() in choice_map:
        matched_choice = choice_map[json_val.lower()]
        if matched_choice in abstain_set:
            return Verdict(correct=False, parsed=matched_choice, parse_failed=False, abstained=True)
        return Verdict(
            correct=(matched_choice == expected),
            parsed=matched_choice,
            parse_failed=False,
            abstained=False,
        )

    # 2. Look for explicit markers like "Answer:", "Final answer:", "Verdict:", etc.
    marker_pattern = r"(?:final\s+answer|answer|verdict|choice|conclusion)\s*:\s*(\*+)?([A-Za-z0-9_-]+)(\*+)?"
    marker_matches = re.findall(marker_pattern, raw, re.IGNORECASE)
    if marker_matches:
        candidate_word = marker_matches[-1][1]
        if candidate_word.lower() in choice_map:
            matched_choice = choice_map[candidate_word.lower()]
            if matched_choice in abstain_set:
                return Verdict(correct=False, parsed=matched_choice, parse_failed=False, abstained=True)
            return Verdict(
                correct=(matched_choice == expected),
                parsed=matched_choice,
                parse_failed=False,
                abstained=False,
            )

    # Word boundary search: tokenization by word breaks would miss multi-word options
    # like "Boston Marathon" (not a single token), causing perfect answers to parse as nothing.
    # This defect would bias prose arms with multi-word options while leaving JSON untouched.
    cleaned = re.sub(r"[*_`]", "", raw).strip()
    occurrences = []
    for choice in choices:
        for m in re.finditer(rf"(?<!\w){re.escape(choice)}(?!\w)", cleaned, re.IGNORECASE):
            occurrences.append((m.start(), m.end(), choice))
    occurrences.sort()

    if not occurrences:
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    def _verdict(choice: str) -> Verdict:
        if choice in abstain_set:
            return Verdict(correct=False, parsed=choice, parse_failed=False, abstained=True)
        return Verdict(correct=(choice == expected), parsed=choice,
                       parse_failed=False, abstained=False)

    distinct = {c for _, _, c in occurrences}
    if len(distinct) == 1:
        return _verdict(occurrences[0][2])

    # Positional disambiguation, applied SYMMETRICALLY so neither answer-first nor
    # answer-last is privileged, avoiding bias in this benchmark comparison.
    # answer-first (A0/A1): the completion opens with an option.
    head = [o for o in occurrences if o[0] == 0]
    if head:
        return _verdict(max(head, key=lambda o: o[1] - o[0])[2])

    # answer-last (A2/A3/A4): the completion closes with an option.
    # The tight window prevents ambiguous phrasings like "either X or Y, hard to say"
    # from silently scoring Y instead of failing to parse.
    tail = [o for o in occurrences if o[1] >= len(cleaned) - 2]
    if tail:
        return _verdict(tail[-1][2])

    return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)
