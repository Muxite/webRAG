"""Citation echo: one URL pasted as the source of N different per-entity claims.

Measured live on 2026-08-28, task 162 -- a 7-branch fan-out asking which STS-61 crew member was
oldest at launch. The run made five real visits, exhausted its visit budget partway through the
fan-out, and completed the remaining branches from parametric memory, attributing all seven
astronauts' dates of birth to the SAME generic NASA landing page. One of those dates was
fabricated (1938-10-29 / age 55 against the real 1935-08-19 / 58) and it is the value the answer
elected as the winner.

Nothing in the engine could see that. Five pages really were opened, so visit-count grounding
passes; the fabricated URL appears once in ``unverified_citations``, a count of ONE that reads
exactly like a run that cited a single page it happened not to open; and no mechanism anywhere
compares the number of per-entity facts asserted against the number of pages actually read.

Two counts, both deterministic and free:

``max_reuse``
    How many DISTINCT per-entity claims attribute to a single URL. Seven-way reuse of one
    landing page across seven different people is the fingerprint of hallucination-fill.

``claims_per_visit``
    Distinct per-entity claims divided by pages opened -- 7/5 = 1.4 on the evidence run. Above
    1.0 a fan-out deliverable asserts more per-entity facts than it opened pages to source them
    from. Cheaper and more robust than URL matching: it survives an answer that cites nothing at
    all, and it catches the sibling failure where each fabricated entity gets its OWN plausible
    unopened URL (max_reuse stays 1). Its known limitation is the mirror image of the echo
    metric's: ONE page can legitimately document many items (a search-results roster, a table),
    so a ratio above 1.0 is a question to ask of a run, not an answer about it.

Reuse is a SIGNAL, NOT PROOF. Two people genuinely documented on one page (a joint biography, a
mission crew roster) produce real reuse from real evidence, which is why this module reports the
number and the entities behind it rather than a verdict, and why ``over_asserted`` -- the ratio
test, which that legitimate case passes -- is kept as a separate property.

Detection is unconditional: ``attach_citation_echo`` stamps the payload whatever
``FinalConfig.citation_echo_enforcement_enabled`` says, so historical captures and future runs
both carry the signal and the false-positive rate is countable before anything acts on it. That
flag is default OFF and reaches no enforcement path yet: this ships as pure telemetry, and no
score, gate or verdict is derived from any field here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.app.idea_policies.candidate_coverage import _extract_name, enumerated_items

_logger = logging.getLogger(__name__)

#: Payload key carrying the whole audit, written whenever the audit is ACTIVE.
CITATION_ECHO = "citation_echo"


def _cite_key(url: str) -> str:
    # Local import: ``idea_finalize`` sits above the policy package and imports from it.
    from agent.app.idea_finalize import _norm_cite

    return _norm_cite(url)


def _cited_urls(text: str) -> List[str]:
    from agent.app.idea_finalize import _URL_RE, _clean_url

    seen: set = set()
    out: List[str] = []
    for match in _URL_RE.findall(text):
        cleaned = _clean_url(match)
        key = _cite_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


@dataclass(frozen=True)
class EntityClaim:
    """One enumerated per-entity assertion: who it is about, and what it cites."""

    entity: str
    urls: Tuple[str, ...]


@dataclass(frozen=True)
class EchoedUrl:
    """One URL and the distinct entity claims that attribute to it."""

    url: str
    claim_count: int
    entities: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "claim_count": self.claim_count,
            "entities": list(self.entities),
        }


@dataclass
class CitationEchoResult:
    """How concentrated a deliverable's per-entity sourcing is, and how thin its reading was.

    Inspectable rather than a bare bool: the diagnostic value is in WHICH URL carried how many
    entities, and in the ratio between asserted facts and opened pages.
    """

    active: bool = False
    distinct_claims: int = 0
    distinct_urls: int = 0
    max_reuse: int = 0
    echoed_urls: List[EchoedUrl] = field(default_factory=list)
    #: Pages the run opened, or ``None`` when the count is unknown/unusable -- never 0 as a
    #: stand-in for "not recorded".
    visits: Optional[int] = None
    reason: str = ""

    @property
    def echoed(self) -> bool:
        """Some URL is cited for two or more DIFFERENT entities. A signal, not proof."""
        return self.max_reuse >= 2

    @property
    def claims_per_visit(self) -> Optional[float]:
        """Distinct per-entity claims per opened page, or ``None`` when visits are unknown/0."""
        if not self.active or not self.visits:
            return None
        return round(self.distinct_claims / self.visits, 3)

    @property
    def over_asserted(self) -> bool:
        """The deliverable asserts more per-entity facts than it opened pages."""
        ratio = self.claims_per_visit
        return ratio is not None and ratio > 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "max_reuse": self.max_reuse,
            "echoed_urls": [e.as_dict() for e in self.echoed_urls],
            "distinct_urls": self.distinct_urls,
            "distinct_claims": self.distinct_claims,
            "visits": self.visits,
            "claims_per_visit": self.claims_per_visit,
            "reason": self.reason,
        }


def entity_claims(deliverable: Any) -> List[EntityClaim]:
    """The per-entity claims an enumerated deliverable makes, one per distinct entity.

    :param deliverable: the final answer text (any non-string fails open).
    :returns: one :class:`EntityClaim` per distinct entity named by the deliverable's
        enumerated run, in first-mention order, carrying the URLs cited on that entity's own
        lines. Repeat mentions of one entity merge into a single claim, so a two-line entry
        cannot inflate the reuse count.
    :raises: nothing — an unparseable or non-enumerated deliverable yields ``[]``.
    """
    ordered: List[str] = []
    urls_by_entity: Dict[str, List[str]] = {}
    for body in enumerated_items(deliverable):
        entity = _extract_name(body)
        if not entity:
            continue
        if entity not in urls_by_entity:
            ordered.append(entity)
            urls_by_entity[entity] = []
        for url in _cited_urls(body):
            if url not in urls_by_entity[entity]:
                urls_by_entity[entity].append(url)
    return [EntityClaim(entity=e, urls=tuple(urls_by_entity[e])) for e in ordered]


def _visit_count(payload: Mapping[str, Any], visits: Any) -> Optional[int]:
    if visits is not None:
        return visits if isinstance(visits, int) and not isinstance(visits, bool) and visits >= 0 else None
    sources = payload.get("sources")
    return len(sources) if isinstance(sources, Sequence) and not isinstance(sources, str) else None


def audit_citation_echo(payload: Any, visits: Any = None) -> CitationEchoResult:
    """How many distinct per-entity claims lean on each cited URL, and on each opened page.

    :param payload: the final payload (``final_deliverable`` + ``sources``). Anything else —
        ``None``, a list, a string, a dict with the wrong field types — is inert.
    :param visits: pages actually opened, when the caller has a better count than
        ``len(payload["sources"])`` (the engine's ``visit.count``). A negative or non-integer
        count is treated as unknown rather than as zero.
    :returns: a :class:`CitationEchoResult`, ``active`` only when the deliverable carries an
        enumerated run of two or more items from which at least one entity name parses (a
        two-line entry about ONE person counts once, so it cannot inflate reuse). Reuse is
        reported, never judged: a page
        genuinely documenting several entities echoes exactly like a fabricated one.
    :raises: nothing — an audit must never crash finalize.
    """
    mapping = payload if isinstance(payload, Mapping) else {}
    counted = _visit_count(mapping, visits)
    try:
        claims = entity_claims(mapping.get("final_deliverable"))
    except Exception as exc:  # noqa: BLE001 — telemetry must never crash finalize
        _logger.warning(f"[CITATION-ECHO] claim parse failed: {exc}")
        return CitationEchoResult(visits=counted, reason="claim parse failed")
    if not claims:
        return CitationEchoResult(
            visits=counted,
            reason="the deliverable makes no enumerated per-entity claims",
        )

    entities_by_url: Dict[str, List[str]] = {}
    url_display: Dict[str, str] = {}
    for claim in claims:
        for url in claim.urls:
            key = _cite_key(url)
            url_display.setdefault(key, url)
            entities_by_url.setdefault(key, []).append(claim.entity)

    echoed = [
        EchoedUrl(url=url_display[key], claim_count=len(entities), entities=tuple(entities))
        for key, entities in entities_by_url.items()
        if len(entities) >= 2
    ]
    echoed.sort(key=lambda e: (-e.claim_count, e.url))
    max_reuse = max((len(e) for e in entities_by_url.values()), default=0)
    return CitationEchoResult(
        active=True,
        distinct_claims=len(claims),
        distinct_urls=len(entities_by_url),
        max_reuse=max_reuse,
        echoed_urls=echoed,
        visits=counted,
        reason=(
            f"{len(claims)} per-entity claim(s) across {len(entities_by_url)} cited URL(s); "
            f"heaviest URL carries {max_reuse}"
        ),
    )


def attach_citation_echo(payload: Any, visits: Any = None) -> None:
    """Stamp ``payload[CITATION_ECHO]`` with the audit, in place, when it is active.

    :param payload: the final payload; mutated. A non-dict is left alone.
    :param visits: as :func:`audit_citation_echo`.
    :returns: ``None``. Nothing else in the payload is read or written, so every score, gate
        and verdict is byte-identical with or without this call.
    :raises: nothing — telemetry must never crash finalize.
    """
    if not isinstance(payload, dict):
        return
    try:
        result = audit_citation_echo(payload, visits)
    except Exception as exc:  # noqa: BLE001 — telemetry must never crash finalize
        _logger.warning(f"[CITATION-ECHO] audit failed: {exc}")
        return
    if not result.active:
        return
    payload[CITATION_ECHO] = result.as_dict()
    if result.over_asserted or result.max_reuse >= 3:
        _logger.warning(
            f"[CITATION-ECHO] {result.distinct_claims} per-entity claim(s), "
            f"{result.visits} page(s) opened, heaviest URL cited for {result.max_reuse}"
        )
