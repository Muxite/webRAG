import dataclasses
import json
import re
from typing import Any, List, Optional, Sequence
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
    # Ignore scheme (keep empty or standard), lower case netloc and path, strip trailing slash from path, drop params, query, fragment
    netloc = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/")
    # urlunparse format: (scheme, netloc, path, params, query, fragment)
    # Using empty scheme so comparison ignores scheme
    return urlunparse(("", netloc, path, "", "", ""))


def grade_url(raw: str, expected_url: str) -> Verdict:
    if not raw or not raw.strip():
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    # Find URL in raw text
    # Match standard http/https URLs
    url_pattern = r'https?://[^\s<>"\')]+'
    matches = re.findall(url_pattern, raw)
    if not matches:
        return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)

    # Use the last URL found or the one marked
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


def _extract_from_json(raw: str) -> Optional[str]:
    # Try parsing whole string or fenced json / json substring
    # Check for fenced code block ```json ... ``` or ``` ... ```
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = fenced if fenced else []

    # Also look for standalone JSON objects { ... }
    json_objs = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    candidates.extend(json_objs)

    for cand in candidates:
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
    # We strip markdown formatting like ** around words.
    marker_pattern = r"(?:final\s+answer|answer|verdict|choice|conclusion)\s*:\s*(\*+)?([A-Za-z0-9_-]+)(\*+)?"
    marker_matches = re.findall(marker_pattern, raw, re.IGNORECASE)
    if marker_matches:
        # Take the last marker match
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

    # 3. Locate every occurrence of every choice, as a WHOLE STRING.
    #
    # Word-tokenising first (the original approach) silently cannot see a
    # multi-word option: "Boston Marathon" is never a single \w+ token, so a
    # completion that answered it perfectly parsed as nothing at all. That
    # defect fell entirely on the prose arms of families with multi-word
    # options while leaving JSON arms untouched, which is a difference between
    # graders masquerading as a difference between prompt shapes.
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

    # Unambiguous: only one distinct option is mentioned anywhere.
    distinct = {c for _, _, c in occurrences}
    if len(distinct) == 1:
        return _verdict(occurrences[0][2])

    # 4. Positional disambiguation, applied SYMMETRICALLY so that neither
    #    answer-first nor answer-last is privileged -- privileging either would
    #    bias the very comparison this benchmark exists to make.
    #
    #    answer-first (A0/A1): the completion opens with an option, and any
    #    other option appears later, inside the justification.
    head = [o for o in occurrences if o[0] == 0]
    if head:
        return _verdict(max(head, key=lambda o: o[1] - o[0])[2])

    #    answer-last (A2/A3/A4): the completion CLOSES with an option. The
    #    window is tight on purpose -- "either SUPPORTED or REFUTED, hard to
    #    say" must stay a parse failure rather than silently scoring REFUTED.
    tail = [o for o in occurrences if o[1] >= len(cleaned) - 2]
    if tail:
        return _verdict(tail[-1][2])

    # Genuinely ambiguous: several options, none in an answer position.
    return Verdict(correct=False, parsed=None, parse_failed=True, abstained=False)
