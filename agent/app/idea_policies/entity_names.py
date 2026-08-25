"""Shallow, model-free named-entity extraction shared by identity guards.

Extracted from ``VisitLeafAction`` (the origin of ``visit_url_identity_guard``) so a second
guard site -- ``idea_sequencing.reorder_for_sequential`` -- can reuse the exact same
name-extraction and sibling-disambiguation logic instead of reimplementing it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.idea_policies.base import DetailKey

#: Capitalised words that describe the ACT of visiting rather than WHICH page -- the leading
#: verb of nearly every planner-written visit title, plus the page-type nouns.
_GENERIC_TITLE_WORDS = frozenset({
    "Visit", "Open", "Read", "Fetch", "Retrieve", "Extract", "Check", "Find", "Search",
    "Page", "Pages", "Site", "Website", "Article", "Wikipedia", "Wiki", "Source", "Sources",
})
_CAPITALISED_WORD_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")


def named_entities(*texts: Optional[str]) -> List[str]:
    """The things a piece of text NAMES, as written: capitalised words minus the generic verbs.

    Deliberately shallow -- capitalisation is the one signal available without a model, and
    the only question asked of it is "does this text name anything at all". A leaf that names
    Suez Canal is one whose grounding can be checked against a URL; a leaf titled "visit a
    source page" names nothing and no check is possible. Never used to decide which page is
    RIGHT, only to reject candidates that mention none of the text's own names.
    """
    names: List[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        for word in _CAPITALISED_WORD_RE.findall(text):
            if word not in _GENERIC_TITLE_WORDS and word not in names:
                names.append(word)
    return names


def distinguishing_names(graph: "IdeaDag", node: "IdeaNode", parent: Optional["IdeaNode"]) -> List[str]:
    """The names that tell THIS node apart from its siblings -- its own, minus the shared ones.

    A breadth fan-out names one entity per arm and repeats the category word in every title
    ("the Suez Canal page", "the Erie Canal page"), so "does this candidate mention a name the
    node uses" is satisfied by ``/wiki/Erie_Canal`` for the Suez node -- the exact confusion
    the guard exists to catch. Dropping the names the siblings ALSO use leaves ``Suez``, which
    the wrong candidate does not mention.

    Empty when the node names nothing, and empty when nothing separates it from a sibling (two
    identically-titled nodes): both mean the node cannot be told apart this way, and the guard
    declines to judge rather than guessing in the other direction.
    """
    named = named_entities(node.title, node.details.get(DetailKey.INTENT.value))
    if not named or parent is None:
        return named
    shared = set()
    for sibling_id in parent.children:
        if sibling_id == node.node_id:
            continue
        sibling = graph.get_node(sibling_id)
        if sibling is None:
            continue
        for name in named_entities(sibling.title, sibling.details.get(DetailKey.INTENT.value)):
            shared.add(name.lower())
    return [name for name in named if name.lower() not in shared]
