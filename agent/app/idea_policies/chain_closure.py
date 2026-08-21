"""Relation-typed backward chain closure: does hop k's page name hop k-1's entity, via the
relation the mandate actually asked about?

A real strong agent solving test 065 (a 3-hop dependent chain: collection -> poet -> that
poet's BIRTHPLACE town -> that town's elevation) hit, and deliberately refused, a real
chain-poisoning decoy: an early search snippet called Temuco Neruda's "native town". Trusting
it would have made hop 3 execute FLAWLESSLY against the WRONG page -- Temuco's own Wikipedia
infobox states a real elevation (360 m), so a visit really happens, visit-count grounding
passes, the page-identity guard passes (the page IS the page the leaf asked for), and even
``grounding.answer_numeric_provenance`` passes, because 360 m genuinely appears in genuinely
fetched text. **The error lives in the RELATION BETWEEN hop 2 and hop 3, not in any single
hop's execution**, and no other mechanism in this engine can see it.

What the strong agent did instead was cheap and mechanical: it confirmed that the NEXT-hop
page itself back-references the PREVIOUS-hop entity through the SAME relation the mandate
named -- Parral's page reads "Parral ... is the birthplace of poet Pablo Neruda". That
back-reference is the closure. Verified live against all three real candidate pages:

    Parral, Chile        "is the birthplace of ... Pablo Neruda"   -> closes
    Temuco               names Neruda only as "lived in" / notable -> does NOT close (the decoy)
    Hidalgo del Parral   never mentions Neruda at all              -> does NOT close (the homonym)

This module reproduces that test deterministically. No LLM call, no new schema field, no new
prompt instruction: the relation cue is mined from the mandate's own wording, the entities from
the visited pages' own identities, and the closure from the pages' own raw text.

Design notes
------------
* **Direction matters.** Only "a LATER page states an EARLIER page's entity near the relation
  cue" counts. The reverse direction is nearly free -- the poet's own page says "Born ...
  Parral, Chile" whatever town the run went on to open -- so accepting it would certify the
  decoy as readily as the truth.

* **Existence, not per-hop bookkeeping.** The audit asks whether ANY ordered pair of visited
  pages closes, rather than pairing hop k with hop k-1 by index. Nothing in the graph records a
  reliable per-visit hop ORDINAL (``waypoint``'s ``step`` is written only under a flag that is
  off by default), and a run interleaves searches, retries and revisits, so an index-paired
  check would fail on visit bookkeeping rather than on chain drift. The existence form still
  rejects both wrong-entity cases above, because neither wrong page back-references the poet
  through the asked-about relation at all.

* **Enforcement is borrowed, not built.** A failed closure becomes a ``missing_requirements``
  entry on the merge, which the existing internal-consistency guard already turns into a
  not-achieved verdict -- and a not-achieved merge is exactly what makes the engine re-plan the
  branch (``merge_incomplete``/``merge_should_skip``), so the retry effect comes for free
  without a second, parallel enforcement path. Same route as
  ``alternative_branch.race_value_agreement`` and ``candidate_roster``.

* **Detection is unconditional, acting on it is gated** (``MergeConfig.chain_closure_enabled``,
  default OFF). The false-positive rate is genuinely unmeasured: a correct chain hop's page need
  not mention the entity that led to it (many town pages list no notable residents at all), and
  the relation vocabulary below is a small curated list. So the audit is stamped on the merge
  node whatever the flag says, which is what makes that rate measurable from report captures
  before anyone turns the downgrade on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.grounding import _norm
from agent.app.idea_policies.shape_classifier import classify_shape

# The same window ``waypoint`` uses for "this text is ABOUT that cue", reused rather than
# re-picked: a back-reference sentence ("Parral ... is the birthplace of poet Pablo Neruda")
# and an infobox label/value pair both sit well inside it, while a page that merely mentions
# the entity somewhere else entirely (Temuco's "notable people" list) does not.
from agent.app.idea_policies.waypoint import _WINDOW, _page_identity

#: Merge-node key carrying the whole audit, written whenever the audit is ACTIVE (closed or
#: not) so the live rate of both outcomes is countable.
CHAIN_CLOSURE = "chain_closure"
#: Merge-node marker written only when an active audit found NO closing pair -- the actionable
#: shape (suspected chain drift).
CHAIN_CLOSURE_OPEN_MARKER = "chain_closure_open"

#: Relations a chain mandate can hop ON. ``(name, mandate_markers, page_cues)``: a relation is
#: ACTIVE only when the mandate's own wording names it, and then its page cues are what a
#: back-reference must sit near in the later page's text. Deliberately small and concrete --
#: every entry is a relation that identifies ONE entity from another (a birthplace, a founder,
#: a designer), which is the only kind of hop a back-reference can close. Generic sequencing
#: language ("then", "that page") names no relation and so activates nothing, which is how a
#: chain mandate with no extractable relation stays inert instead of guessing.
_RELATIONS: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "birthplace",
        ("birthplace", "born in", "was born", "were born", "place of birth", "native town",
         "birth town", "hometown", "home town"),
        ("birthplace", "born in", "was born", "born at", "place of birth", "native of",
         "birthplace of", "hometown of"),
    ),
    (
        "deathplace",
        ("died in", "place of death", "death place", "they died", "was assassinated"),
        ("died in", "died at", "place of death", "death of", "assassinated in"),
    ),
    (
        "founder",
        ("founded by", "founder of", "who founded", "its founder", "was founded"),
        ("founded by", "founder", "was founded", "co-founded by"),
    ),
    (
        "designer",
        ("designed by", "designer of", "who designed", "architect of", "chief engineer",
         "engineered by", "who built", "built by"),
        ("designed by", "designer", "architect", "engineered by", "chief engineer",
         "built by", "constructed by"),
    ),
    (
        "author",
        ("written by", "author of", "who wrote", "wrote the", "poet who", "composer of",
         "who composed"),
        ("written by", "author of", "wrote", "poet", "novel by", "composed by", "composer"),
    ),
    (
        "discoverer",
        ("discovered by", "who discovered", "invented by", "who invented", "named after"),
        ("discovered by", "discoverer", "invented by", "inventor", "named after",
         "named in honour of", "named in honor of"),
    ),
    (
        "location",
        ("located in", "situated in", "headquartered in", "headquarters of", "capital of",
         "capital city of", "which city", "which town"),
        ("located in", "situated in", "headquarters", "capital of", "capital city",
         "is a city in", "is a town in"),
    ),
)

#: Identity tokens with no discriminating power: URL/site furniture plus the generic nouns a
#: page uses about ITSELF ("the town was founded in 1881"), which would otherwise let any page
#: close on any other page whose title contains a common category word.
_GENERIC_TOKENS = frozenset({
    "http", "https", "www", "wiki", "wikipedia", "html", "index", "page", "wikimedia",
    "article", "encyclopedia", "free",
    "city", "town", "village", "commune", "county", "province", "region", "state",
    "district", "municipality", "department", "prefecture", "country", "national",
    "list", "history", "people", "notable", "about",
})

#: Identity tokens must be this long: shorter ones are prepositions, articles and initials that
#: match everywhere ("del", "of", "the", "san" is borderline and deliberately excluded).
_MIN_TOKEN = 4

_TOKEN_RE = re.compile(r"[a-z0-9]{%d,}" % _MIN_TOKEN)
#: A title's LEADING segment: "Parral, Chile" -> "Parral", "Temuco (Chile)" -> "Temuco". A
#: qualifier after the comma is the country/region the title disambiguates WITH, and letting
#: "chile" act as an entity token would close a chain on any Chilean page at all.
_LEAD_SPLIT_RE = re.compile(r"[,(\[–—]|\s[-−]\s")


@dataclass(frozen=True)
class VisitedPage:
    """One distinct successfully-visited page: its URL, its identity tokens, its raw text."""

    url: str
    identity: str
    tokens: Tuple[str, ...]
    text: str


@dataclass
class ChainClosureResult:
    """Whether a chain-shaped run's hops are relation-linked to each other.

    Inspectable rather than a bare bool: the diagnostic value is in WHICH relation was asked
    about and WHICH pages failed to reference each other through it.
    """

    active: bool = False
    closed: bool = False
    relation: str = ""
    relation_cues: Tuple[str, ...] = ()
    pages: List[str] = field(default_factory=list)
    #: ``(earlier page url, later page url, the entity token that closed, the cue it sat near)``
    closure: Optional[Tuple[str, str, str, str]] = None
    reason: str = ""

    @property
    def drifted(self) -> bool:
        """The actionable shape: the audit applied and found no relation-linked pair."""
        return self.active and not self.closed

    def missing_requirements(self) -> List[str]:
        if not self.drifted:
            return []
        return [
            f"no page opened in this chain states a previous hop's entity near the mandate's "
            f"'{self.relation}' relation, so the hand-off from one hop to the next is "
            f"unverified (pages: {', '.join(self.pages)})"
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "closed": self.closed,
            "relation": self.relation,
            "relation_cues": list(self.relation_cues),
            "pages": list(self.pages),
            "closure": list(self.closure) if self.closure else None,
            "reason": self.reason,
        }


def mandate_relation(mandate: Any) -> Tuple[str, Tuple[str, ...]]:
    """The relation the mandate hops on, as ``(name, page cues)``, or ``("", ())``.

    The FIRST relation in ``_RELATIONS`` whose mandate markers appear wins, and only that one:
    a mandate mentioning two relations is hopping on one of them and guessing the union would
    only widen what counts as a closure.
    """
    text = str(mandate or "").lower()
    if not text:
        return "", ()
    for name, markers, cues in _RELATIONS:
        if any(m in text for m in markers):
            return name, cues
    return "", ()


def _root_mandate(graph) -> str:
    """The whole task's mandate, off the root node.

    Read from the ROOT rather than from a merge's own ``original_goal``: the chain shape is a
    property of the task, and a nested merge's resolved goal is a sub-goal that names neither
    the chain nor its relation.
    """
    try:
        root = graph.get_node(graph.root_id())
    except Exception:  # noqa: BLE001 -- an audit must never crash a merge
        return ""
    if root is None:
        return ""
    details = root.details if isinstance(getattr(root, "details", None), dict) else {}
    for key in (DetailKey.GOAL.value, DetailKey.ORIGINAL_GOAL.value, "mandate"):
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(getattr(root, "title", "") or "").strip()


def _identity_text(result: Dict[str, Any]) -> str:
    """The page's own name, preferring a title over the URL.

    Falls back to ``waypoint._page_identity`` (title + h1 + url, never body text) so a result
    carrying only a URL still yields tokens from its slug; the generic-token filter drops the
    site furniture that comes with it.
    """
    for key in ("page_title", "h1_text", "title"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return unquote(_page_identity(result)).replace("_", " ").replace("/", " ")


def _entity_tokens(identity: str) -> Tuple[str, ...]:
    lead = _LEAD_SPLIT_RE.split(identity, maxsplit=1)[0]
    tokens = [t for t in _TOKEN_RE.findall(lead.lower()) if t not in _GENERIC_TOKENS]
    # Dedup, order preserved, so the reported closing token is the page's own leading name.
    seen, out = set(), []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return tuple(out)


def _page_text(result: Dict[str, Any]) -> str:
    from agent.app.idea_policies.action_constants import ActionResultKey

    for key in (
        ActionResultKey.CONTENT_FULL.value,
        ActionResultKey.CONTENT.value,
        ActionResultKey.CONTENT_WITH_LINKS.value,
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def visited_pages(graph) -> List[VisitedPage]:
    """Every distinct successfully-visited page, in graph (dependency) order.

    De-duplicated by normalized URL keeping the FIRST occurrence, so a revisit cannot reorder
    the chain. Search results are excluded by construction (their action is not ``visit``) --
    the same "a snippet is not a page you read" line every other check in this family draws,
    and the exact line the decoy in this module's docstring crossed.
    """
    pages: List[VisitedPage] = []
    seen = set()
    for node in graph.iter_depth_first():
        details = getattr(node, "details", {}) or {}
        result = details.get(DetailKey.ACTION_RESULT.value) or {}
        if not (
            isinstance(result, dict)
            and result.get("action") == IdeaActionType.VISIT.value
            and result.get("success")
        ):
            continue
        url = str(result.get("url") or result.get("source_url") or "")
        key = _norm(url) or f"#{len(pages)}"
        if key in seen:
            continue
        seen.add(key)
        identity = _identity_text(result)
        pages.append(
            VisitedPage(
                url=url or identity,
                identity=identity,
                tokens=_entity_tokens(identity),
                text=_page_text(result),
            )
        )
    return pages


def _closes_on(text: str, tokens: Sequence[str], cues: Sequence[str]) -> Optional[Tuple[str, str]]:
    """``(token, cue)`` when ``text`` states one of ``tokens`` within ``_WINDOW`` characters of
    one of ``cues``, else ``None``. Every cue OCCURRENCE is examined, not just the first: a
    page's own lead sentence and its infobox both carry the relation word, and only one of them
    needs the back-reference.
    """
    if not text or not tokens or not cues:
        return None
    low = text.lower()
    token_res = [(tok, re.compile(rf"\b{re.escape(tok)}\b")) for tok in tokens]
    for cue in cues:
        for match in re.finditer(re.escape(cue), low):
            window = low[max(0, match.start() - _WINDOW): match.end() + _WINDOW]
            for tok, token_re in token_res:
                if token_re.search(window):
                    return tok, cue
    return None


def audit_chain_closure(graph, mandate: Optional[str] = None) -> ChainClosureResult:
    """Is this chain-shaped run's hop hand-off backed by a relation-typed back-reference?

    Inert (``active=False``) unless all of the following hold, each an independent reason the
    question would be meaningless rather than answered negatively:

    * the mandate classifies as a ``chain`` (``shape_classifier.classify_shape``);
    * the mandate names one of the ``_RELATIONS`` above;
    * the run fetched at least two distinct pages, one of which carries usable entity tokens
      and a later one of which carries text.

    Never raises: the worst outcome is an inactive result with a reason.
    """
    text = mandate if isinstance(mandate, str) and mandate.strip() else _root_mandate(graph)
    if classify_shape(text or "") != "chain":
        return ChainClosureResult(reason="mandate is not chain-shaped")
    relation, cues = mandate_relation(text)
    if not relation:
        return ChainClosureResult(reason="mandate names no hop relation")

    pages = visited_pages(graph)
    if len(pages) < 2:
        return ChainClosureResult(
            relation=relation, relation_cues=cues,
            pages=[p.url for p in pages],
            reason="fewer than two distinct pages were fetched",
        )

    urls = [p.url for p in pages]
    checkable = False
    for i, earlier in enumerate(pages):
        if not earlier.tokens:
            continue
        for later in pages[i + 1:]:
            if not later.text:
                continue
            checkable = True
            hit = _closes_on(later.text, earlier.tokens, cues)
            if hit:
                token, cue = hit
                return ChainClosureResult(
                    active=True, closed=True, relation=relation, relation_cues=cues,
                    pages=urls, closure=(earlier.url, later.url, token, cue),
                    reason=(
                        f"{later.url} states '{token}' near '{cue}' -- the hop from "
                        f"{earlier.url} is relation-linked"
                    ),
                )
    if not checkable:
        return ChainClosureResult(
            relation=relation, relation_cues=cues, pages=urls,
            reason="no ordered page pair carried both an identity and page text",
        )
    return ChainClosureResult(
        active=True, closed=False, relation=relation, relation_cues=cues, pages=urls,
        reason=(
            f"no fetched page states an earlier page's entity near the mandate's "
            f"'{relation}' relation"
        ),
    )
