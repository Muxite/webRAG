"""Structured records for what a VISIT actually brought back (observe-only).

A completed VISIT stores its page as a raw prose blob (``content`` / ``content_full`` /
``content_with_links``). Nothing downstream ever parses that blob, so "which page said
this, and what exactly did it say" is only ever re-derived by an LLM reading the prose
again. This module is the first structured view of that:

* :class:`Evidence` -- one visited page, deterministic and free. No LLM, no network: it
  is a projection of the ``action_result`` dict the VISIT already produced.
* :class:`Claim` -- one ``(subject, predicate, value)`` triple read off ONE Evidence's
  excerpt by ONE cheap LLM call. 1:1 with its Evidence on purpose: cross-evidence claim
  merging and verification are later work, so every claim here is born ``"unverified"``
  and nothing ever moves it off that state yet.

Both are SIDECAR data. Gated by ``RunPolicy.evidence_store_mode`` (``"off"`` by default,
``"observe"`` to record), attached to a node's ``details`` under
:attr:`~agent.app.idea_policies.base.DetailKey.EVIDENCE` /
:attr:`~agent.app.idea_policies.base.DetailKey.CLAIMS`, and read by nothing: they do not
touch ``action_result``, scoring, ``success``/``finalization_status``, or any prompt.

Judgment calls made here rather than guessed silently past
-----------------------------------------------------------
* ``source_type`` is NOT a real classifier. A domain pattern cannot tell a primary source
  from a secondary one, and pretending otherwise would put an unmeasured label on every
  record. So: a small, unambiguous encyclopedia/reference host set maps to ``"reference"``
  and everything else to ``"unknown"``. ``"primary"`` is a declared value that this
  heuristic NEVER emits -- it is reserved for the classifier that earns it.
* ``canonical_url`` is local rather than one of the two existing normalizers
  (``idea_test_utils.normalize_url``, ``promptbench.grade.normalize_url``). Both of those
  build MATCHING KEYS: they drop the scheme entirely and the promptbench one drops the
  query string too. An Evidence's canonical URL has to stay a real, fetchable address of
  the same page, so this one lowercases scheme+host, strips the fragment and a trailing
  slash, and keeps the query.
* ``EXCERPT_CHARS`` is 500: half of ``ExpansionConfig.ancestor_content_chars`` (1000), the
  established cap for "how much page text is worth carrying into one prompt". Half,
  because the claim call fires once per visited page and is pure telemetry -- it should
  not cost a fraction of an expansion prompt, let alone a multiple of one.
* ``MAX_CLAIMS`` is 8, enforced after parsing rather than via the JSON schema (nothing in
  ``idea_dag_schemas`` uses ``maxItems``; see the note on ``CLAIM_EXTRACTION_JSON_SCHEMA``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from agent.app.idea_dag_schemas import CLAIM_EXTRACTION_JSON_SCHEMA
from agent.app.idea_policies.action_constants import ActionResultKey, PromptBuilder
from agent.app.idea_policies.json_repair import repair_malformed_json

_logger = logging.getLogger(__name__)

#: Page text carried into the claim-extraction prompt. See the module docstring.
EXCERPT_CHARS = 500

#: Hard cap on claims kept from one page, applied after parsing.
MAX_CLAIMS = 8

#: Response budget for the claim call. ``MAX_CLAIMS`` short triples fit comfortably; the
#: point of the bound is that an observer must not be able to run up a long generation.
CLAIM_MAX_TOKENS = 400

#: Hosts that are unambiguously tertiary/reference works. Everything else stays
#: ``"unknown"`` -- see the module docstring on why this is not a real classifier.
_REFERENCE_HOST_MARKERS = (
    "wikipedia.org",
    "wikidata.org",
    "wiktionary.org",
    "wikimedia.org",
    "britannica.com",
)

SOURCE_TYPE_PRIMARY = "primary"
SOURCE_TYPE_REFERENCE = "reference"
SOURCE_TYPE_UNKNOWN = "unknown"

VERIFICATION_UNVERIFIED = "unverified"
VERIFICATION_SUPPORTED = "supported"
VERIFICATION_CONTRADICTED = "contradicted"

#: Telemetry/log site name shared by the claim call and its JSON repair.
_CLAIM_SITE = "evidence_claim_extraction"


def canonicalize_url(url: Any) -> str:
    """Lowercase scheme+host, drop the fragment, strip one trailing slash; keep the query.

    Never raises: an unparseable or non-string input degrades to the stripped raw text.
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if not parts.netloc:
        # Scheme-less or otherwise unsplittable ("example.com/x"): normalize what we can
        # without inventing a scheme the caller never fetched.
        return raw.split("#", 1)[0].rstrip("/")
    path = parts.path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )


def classify_source_type(canonical_url: str) -> str:
    """``"reference"`` for a known reference host, else ``"unknown"``. Never ``"primary"``."""
    host = ""
    try:
        host = urlsplit(canonical_url or "").netloc.lower()
    except ValueError:
        host = ""
    if not host:
        host = str(canonical_url or "").lower()
    if any(marker in host for marker in _REFERENCE_HOST_MARKERS):
        return SOURCE_TYPE_REFERENCE
    return SOURCE_TYPE_UNKNOWN


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Evidence:
    """One successfully visited page, as a structured record rather than a prose blob."""

    id: str
    url: str
    canonical_url: str
    title: str
    source_type: str
    excerpt: str
    fetched_at: float
    node_id: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot, suitable for a node ``details`` value."""
        return {
            "id": self.id,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "source_type": self.source_type,
            "excerpt": self.excerpt,
            "fetched_at": self.fetched_at,
            "node_id": self.node_id,
        }


@dataclass(frozen=True)
class Claim:
    """One ``(subject, predicate, value)`` triple attributed to exactly one Evidence."""

    id: str
    subject: str
    predicate: str
    value: str
    evidence_id: str
    verification_state: str = VERIFICATION_UNVERIFIED

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot, suitable for a node ``details`` value."""
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "evidence_id": self.evidence_id,
            "verification_state": self.verification_state,
        }


def extract_evidence(node: Any, action_result: Dict[str, Any]) -> Evidence:
    """Project a completed VISIT's ``action_result`` into an :class:`Evidence`.

    Pure, deterministic, free. Must never raise: this runs on the success path of a VISIT
    that has already worked, so a missing title / absent timestamp / non-dict result
    degrades to an empty-ish record rather than breaking the node.
    """
    result = action_result if isinstance(action_result, dict) else {}
    node_id = _as_text(getattr(node, "node_id", None) or getattr(node, "id", "") or "")

    url = _as_text(
        result.get(ActionResultKey.URL.value)
        or result.get("source_url")
        or ""
    ).strip()
    canonical = canonicalize_url(url)

    title = _as_text(
        result.get("page_title")
        or result.get("h1_text")
        or getattr(node, "title", "")
        or ""
    ).strip()

    content = _as_text(
        result.get(ActionResultKey.CONTENT.value)
        or result.get(ActionResultKey.CONTENT_FULL.value)
        or ""
    )
    excerpt = content[:EXCERPT_CHARS]

    fetched_at = _as_float(result.get(ActionResultKey.TIMESTAMP.value))
    if fetched_at is None:
        # Run-relative node interval (see ``IdeaDagEngine._stamp_node_end``) is the only
        # other timestamp a visit reliably has; 0.0 when even that is absent.
        fetched_at = _as_float(getattr(node, "ended_at", None))
    if fetched_at is None:
        fetched_at = _as_float(getattr(node, "started_at", None))
    if fetched_at is None:
        fetched_at = 0.0

    # Identity is the page AND the node that fetched it: two nodes visiting the same URL
    # are two pieces of evidence (they can disagree, e.g. across a re-fetch), while one
    # node re-running is the same one.
    digest = hashlib.sha1(f"{canonical}\n{node_id}".encode("utf-8")).hexdigest()[:12]

    return Evidence(
        id=f"ev-{digest}",
        url=url,
        canonical_url=canonical,
        title=title,
        source_type=classify_source_type(canonical),
        excerpt=excerpt,
        fetched_at=fetched_at,
        node_id=node_id,
    )


_CLAIM_SYSTEM_PROMPT = (
    "You extract atomic factual triples from a web page excerpt. Return JSON only: "
    '{"claims": [{"subject": ..., "predicate": ..., "value": ...}]}. '
    f"At most {MAX_CLAIMS} claims, each one short and self-contained. "
    "Copy values from the excerpt verbatim; invent nothing, infer nothing, and omit "
    "anything the excerpt does not state. Return an empty list if it states no facts."
)


def _build_claim_messages(evidence: Evidence) -> List[Dict[str, str]]:
    return PromptBuilder.build_messages(
        system_content=_CLAIM_SYSTEM_PROMPT,
        user_content=(
            f"PAGE TITLE: {evidence.title or '(unknown)'}\n"
            f"URL: {evidence.url}\n\n"
            f"EXCERPT:\n{evidence.excerpt}\n\n"
            "Claims JSON:"
        ),
    )


def _claims_from_payload(data: Any, evidence: Evidence) -> List[Claim]:
    """Turn a parsed response into claims, dropping anything that is not a full triple."""
    if not isinstance(data, dict):
        return []
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        return []
    claims: List[Claim] = []
    for item in raw_claims:
        if len(claims) >= MAX_CLAIMS:
            break
        if not isinstance(item, dict):
            continue
        subject = _as_text(item.get("subject")).strip()
        predicate = _as_text(item.get("predicate")).strip()
        value = _as_text(item.get("value")).strip()
        if not (subject and predicate and value):
            continue
        claims.append(
            Claim(
                id=f"{evidence.id}-c{len(claims)}",
                subject=subject,
                predicate=predicate,
                value=value,
                evidence_id=evidence.id,
            )
        )
    return claims


async def extract_claims(
    evidence: Evidence,
    io: Any,
    *,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = CLAIM_MAX_TOKENS,
    temperature: float = 0.0,
    timeout_seconds: Optional[float] = None,
    fallback_model: Optional[str] = None,
) -> List[Claim]:
    """Read up to :data:`MAX_CLAIMS` triples off ``evidence.excerpt`` with ONE LLM call.

    Fails CLOSED to an empty list and NEVER raises. This is observe-only telemetry
    attached to a VISIT that already succeeded, so an outage, a timeout, a refusal or a
    model that cannot produce JSON must cost the run nothing but the call it already made.
    Malformed JSON gets exactly one shared repair attempt
    (:func:`idea_policies.json_repair.repair_malformed_json`) before giving up.
    """
    if not evidence.excerpt.strip():
        return []
    messages = _build_claim_messages(evidence)
    try:
        payload = io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=CLAIM_EXTRACTION_JSON_SCHEMA,
        )
        response = await io.query_llm_with_fallback(
            payload,
            model_name=model_name,
            fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must never fail its VISIT
        _logger.warning(f"[EVIDENCE] claim extraction call failed: {exc}")
        return []

    if not response:
        return []
    try:
        data = json.loads(response)
    except json.JSONDecodeError as exc:
        try:
            data = await repair_malformed_json(
                io,
                site=_CLAIM_SITE,
                messages=messages,
                malformed_text=response,
                parse_error=exc,
                fallback_model=fallback_model,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                logger=_logger,
            )
        except Exception as repair_exc:  # noqa: BLE001 - repair is best-effort
            _logger.warning(f"[EVIDENCE] claim JSON repair failed: {repair_exc}")
            data = None
        if data is None:
            return []
    try:
        return _claims_from_payload(data, evidence)
    except Exception as exc:  # noqa: BLE001 - a weird payload is not a failed visit
        _logger.warning(f"[EVIDENCE] claim payload was unusable: {exc}")
        return []
