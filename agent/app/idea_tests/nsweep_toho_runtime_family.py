"""
Shared template for the N-SWEEP task family (tests 165 / 166 / 167 / 168).

ONE task template, ONE variable: the roster size N in {4, 8, 16, 32}. Everything else -- the
domain, the roster page, the per-item page type, the extracted field, the aggregate, the
mandate wording, the deliverables, the compiled-plan shape and the validators -- is byte-for-byte
identical across the four tasks, so a score difference between them can only come from N.

WHY THIS SHAPE
    ``execution_sequential`` keeps a FIXED 12-step sliding scratchpad window (``_SCRATCHPAD_WINDOW``)
    inside a 25-step budget, so a linear ReAct loop cannot retain more than ~6 resolved items live.
    Tasks 161-164 tried to probe that ceiling with 6-17 items but confounded per-item difficulty
    with N (every variant scored 0, so no crossover was observable). This family removes the
    confound BY CONSTRUCTION: the four rosters are NESTED PREFIXES of one ordered list, so the
    N=4 roster is a subset of the N=8 roster is a subset of the N=16 roster is a subset of the
    N=32 roster. Item #2 is literally the same page read in all four tasks. Per-item difficulty
    is therefore not merely "matched", it is the same item.

TEMPLATE
    Roster (discovered, not given): the article "List of Godzilla films" numbers the Japanese
    (Toho) films in release order in a "N-mark" column. The mandate names only that page and the
    prefix length N -- no film title ever appears in the mandate or in the compiled plan, so the
    roster must be fetched before any fan-out can start.
    Per item (deliberately trivial): open the film's own Wikipedia article, read ``runtime``
    from the infobox. One page, one unambiguous integer, no arithmetic, no second hop.
    Aggregate (needs the whole roster): which film has the SHORTEST running time (argmin), plus
    the combined total running time (a sum, which is wrong if even one item is missing).

GROUND TRUTH -- every figure read off the en.wikipedia infobox ``runtime`` field via the
MediaWiki API (``action=query&prop=revisions&rvsection=0``), verified 2026-08-30:

     #  film                                              runtime   in N=4/8/16/32
     1  Godzilla (1954 film)                               96 min   4 8 16 32
     2  Godzilla Raids Again                               81 min   4 8 16 32   <- argmin @ N=4,8
     3  King Kong vs. Godzilla                             97 min   4 8 16 32
     4  Mothra vs. Godzilla                                88 min   4 8 16 32   <- runner-up @ N=4
     5  Ghidorah, the Three-Headed Monster                 93 min     8 16 32
     6  Invasion of Astro-Monster                          94 min     8 16 32
     7  Ebirah, Horror of the Deep                         87 min     8 16 32
     8  Son of Godzilla                                    86 min     8 16 32   <- runner-up @ N=8
     9  Destroy All Monsters                               88 min       16 32
    10  All Monsters Attack                                70 min       16 32   <- argmin @ N=16,32
    11  Godzilla vs. Hedorah                               85 min       16 32
    12  Godzilla vs. Gigan                                 89 min       16 32
    13  Godzilla vs. Megalon                               81 min       16 32   <- runner-up @ N=16,32
    14  Godzilla vs. Mechagodzilla                         84 min       16 32
    15  Terror of Mechagodzilla                            83 min       16 32
    16  The Return of Godzilla                            103 min       16 32
    17  Godzilla vs. Biollante                            104 min          32
    18  Godzilla vs. King Ghidorah                        103 min          32
    19  Godzilla vs. Mothra                               102 min          32
    20  Godzilla vs. Mechagodzilla II                     107 min          32
    21  Godzilla vs. SpaceGodzilla                        107 min          32
    22  Godzilla vs. Destoroyah                           103 min          32
    23  Godzilla 2000                                     107 min          32
    24  Godzilla vs. Megaguirus                           105 min          32
    25  Godzilla, Mothra and King Ghidorah: GMAOA         105 min          32
    26  Godzilla Against Mechagodzilla                     88 min          32
    27  Godzilla: Tokyo S.O.S.                             91 min          32
    28  Godzilla: Final Wars                              125 min          32
    29  Shin Godzilla                                     120 min          32
    30  Godzilla: Planet of the Monsters                   88 min          32
    31  Godzilla: City on the Edge of Battle              100 min          32
    32  Godzilla: The Planet Eater                         91 min          32

AGGREGATES AND MARGINS
    N=4   argmin Godzilla Raids Again 81 min; runner-up 88 (#4)  -> margin  7 min (8.6%)
    N=8   argmin Godzilla Raids Again 81 min; runner-up 86 (#8)  -> margin  5 min (6.2%)
    N=16  argmin All Monsters Attack  70 min; runner-up 81 (#2)  -> margin 11 min (15.7%)
    N=32  argmin All Monsters Attack  70 min; runner-up 81 (#2)  -> margin 11 min (15.7%)
    total runtime: 362 (N=4) / 722 (N=8) / 1405 (N=16) / 3051 (N=32) minutes.
    The argmin is unique at every N (81 is shared by #2 and #13, but #13 only enters at N=16,
    where the argmin is already the strictly smaller 70).

DECISION-CRITICAL ITEMS (which forgotten item can change the answer)
    N=4, N=8   : #2 (the argmin). Dropping #2 flips the answer to #4 (N=4) or #8 (N=8).
                 Every other item is argmin-irrelevant -- but all of them move the total.
    N=16, N=32 : #10 (the argmin). Dropping #10 flips the answer to #2 or #13 (both 81).
                 #10 sits in the FIRST THIRD of the N=32 roster, which is exactly the region a
                 12-step sliding scratchpad has evicted by the time it reaches item 32. A run
                 that answers "Godzilla Raids Again" on the N=16/N=32 task is showing the
                 early-item forgetting signature, not a lookup error.
    Non-critical for the argmin at every N: everything at 83 min or above.

SINGLE-MISREAD ROBUSTNESS
    Each of the 32 infoboxes carries exactly ONE runtime value (verified), so the ordinary
    failure mode is a missing item, not a wrong one. The one real flip risk is an agent that
    reports the runtime of a re-edited AMERICAN version discussed in an article's body instead
    of the infobox figure: "Godzilla, King of the Monsters!" (the 1956 US re-edit of #1) runs
    80 minutes, which is below the N=4/N=8 argmin of 81. That is a separate Wikipedia article,
    not #1's infobox, and the mandate closes it explicitly ("use the film's own infobox, not a
    re-edited American version"). The other US cuts (King Kong vs. Godzilla 91, Ghidorah 85,
    Invasion of Astro-Monster 92, Godzilla's Revenge 69 for #10) do not change any argmin.

STABILITY / LEAK RESISTANCE
    All 32 films were released between 1954 and 2018 and their runtimes are settled; #33
    (2023) and #34 (2026) sit outside every roster, so new entries append harmlessly and the
    prefixes stay frozen. The runtimes are page-only integers no cheap model can produce
    parametrically, and the roster page publishes no runtime column, so N page reads are
    genuinely required -- a run cannot shortcut the fan-out from the list page.

SOURCES VERIFIED (live, 2026-08-30)
    https://en.wikipedia.org/wiki/List_of_Godzilla_films
        column headers verbatim: "N-mark | Title | Year | Director(s) | Effects director(s) |
        Monster co-star(s)" -- no runtime/duration column; 34 Toho rows, #1..#32 as tabled above.
    https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvsection=0
        &titles=<the 32 film articles>   -- infobox ``| runtime = NN minutes`` for each.
    Cross-check: Wikidata P2047 disagrees with the en-WP infobox on 9 of the 32 films (e.g.
    it says 100 for #1, 69 for #10, 96 for #11); the en-WP infobox is what an agent reads, so
    every figure above comes from the infobox and Wikidata was discarded.
"""

from typing import Any, Callable, Dict, List
import re

from agent.app.idea_test_utils import extract_final_text


ROSTER_PAGE = "List of Godzilla films"
ROSTER_PAGE_URL = "https://en.wikipedia.org/wiki/List_of_Godzilla_films"

SWEEP_SIZES = (4, 8, 16, 32)

# Distance (characters) allowed between a film's title mention and its runtime figure for the
# per-item coverage diagnostic. Wide enough for a "N. Title (year) - 96 minutes - <url>" table
# row or a prose sentence, tight enough that an adjacent row's figure rarely bleeds in.
_PAIR_WINDOW = 100


def _slug_rx(slug: str) -> str:
    """Build a citation regex for a Wikipedia slug, tolerating percent-encoding.

    :param slug: the article slug exactly as it appears after ``/wiki/`` (e.g.
        ``"Godzilla:_Tokyo_S.O.S."``).
    :returns: a lowercased regex string matching ``wiki/<slug>`` raw or percent-encoded.
    """
    out = ["wiki/"]
    for ch in slug.lower():
        if ch == ":":
            out.append("(?::|%3a)")
        elif ch == ",":
            out.append("(?:,|%2c)")
        elif ch == " ":
            out.append("(?:_|%20|\\s)")
        elif ch in ".()[]{}*+?^$|\\":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _film(index: int, title: str, runtime: int, slug: str, title_rx: str,
          leak_tokens: List[str], slug_rx_suffix: str = "") -> Dict[str, Any]:
    """Assemble one roster entry.

    :param index: 1-based position in the ordered roster (the "N-mark" column).
    :param title: the film's exact English Wikipedia article title.
    :param runtime: the infobox running time in whole minutes.
    :param slug: the article slug after ``/wiki/``.
    :param title_rx: lowercased regex identifying the title in an answer, written so it cannot
        also match a different film on the roster.
    :param leak_tokens: distinctive lowercased substrings that must never appear in the mandate
        or the compiled plan (the generic franchise word "godzilla" is deliberately excluded).
    :param slug_rx_suffix: optional regex appended to the citation pattern to stop a slug that
        is a prefix of another film's slug from matching both.
    :returns: the roster-entry dict.
    """
    return {
        "index": index,
        "title": title,
        "runtime": runtime,
        "slug": slug,
        "url": f"https://en.wikipedia.org/wiki/{slug}",
        "title_rx": title_rx,
        "value_rx": rf"\b{runtime}\b",
        "slug_rx": _slug_rx(slug) + slug_rx_suffix,
        "leak_tokens": leak_tokens,
    }


# The ordered roster, ground truth above. Single source of truth for the mandate's N, the
# validators, the margins and the leak assertions.
FILMS: List[Dict[str, Any]] = [
    _film(1, "Godzilla (1954 film)", 96, "Godzilla_(1954_film)",
          r"godzilla[^a-z0-9]{0,12}1954", ["1954"]),
    _film(2, "Godzilla Raids Again", 81, "Godzilla_Raids_Again",
          r"godzilla\s+raids\s+again", ["raids again"]),
    _film(3, "King Kong vs. Godzilla", 97, "King_Kong_vs._Godzilla",
          r"king\s+kong\s+vs\.?\s+godzilla", ["king kong"]),
    _film(4, "Mothra vs. Godzilla", 88, "Mothra_vs._Godzilla",
          r"mothra\s+vs\.?\s+godzilla", ["mothra vs"]),
    _film(5, "Ghidorah, the Three-Headed Monster", 93, "Ghidorah,_the_Three-Headed_Monster",
          r"ghidorah,?\s+the\s+three[-\s]headed", ["three-headed"]),
    _film(6, "Invasion of Astro-Monster", 94, "Invasion_of_Astro-Monster",
          r"invasion\s+of\s+astro", ["astro-monster"]),
    _film(7, "Ebirah, Horror of the Deep", 87, "Ebirah,_Horror_of_the_Deep",
          r"ebirah", ["ebirah"]),
    _film(8, "Son of Godzilla", 86, "Son_of_Godzilla",
          r"son\s+of\s+godzilla", ["son of godzilla"]),
    _film(9, "Destroy All Monsters", 88, "Destroy_All_Monsters",
          r"destroy\s+all\s+monsters", ["destroy all monsters"]),
    _film(10, "All Monsters Attack", 70, "All_Monsters_Attack",
          r"all\s+monsters\s+attack", ["all monsters attack"]),
    _film(11, "Godzilla vs. Hedorah", 85, "Godzilla_vs._Hedorah",
          r"hedorah", ["hedorah"]),
    _film(12, "Godzilla vs. Gigan", 89, "Godzilla_vs._Gigan",
          r"gigan\b", ["gigan"]),
    _film(13, "Godzilla vs. Megalon", 81, "Godzilla_vs._Megalon",
          r"megalon", ["megalon"]),
    _film(14, "Godzilla vs. Mechagodzilla", 84, "Godzilla_vs._Mechagodzilla",
          r"godzilla\s+vs\.?\s+mechagodzilla(?!\s*(?:ii|2)\b)", ["vs. mechagodzilla"],
          slug_rx_suffix=r"(?!_ii)"),
    _film(15, "Terror of Mechagodzilla", 83, "Terror_of_Mechagodzilla",
          r"terror\s+of\s+mechagodzilla", ["terror of mechagodzilla"]),
    _film(16, "The Return of Godzilla", 103, "The_Return_of_Godzilla",
          r"return\s+of\s+godzilla", ["return of godzilla"]),
    _film(17, "Godzilla vs. Biollante", 104, "Godzilla_vs._Biollante",
          r"biollante", ["biollante"]),
    _film(18, "Godzilla vs. King Ghidorah", 103, "Godzilla_vs._King_Ghidorah",
          r"godzilla\s+vs\.?\s+king\s+ghidorah", ["king ghidorah"]),
    _film(19, "Godzilla vs. Mothra", 102, "Godzilla_vs._Mothra",
          r"godzilla\s+vs\.?\s+mothra", ["vs. mothra"]),
    _film(20, "Godzilla vs. Mechagodzilla II", 107, "Godzilla_vs._Mechagodzilla_II",
          r"godzilla\s+vs\.?\s+mechagodzilla\s*(?:ii|2)\b", ["mechagodzilla ii"]),
    _film(21, "Godzilla vs. SpaceGodzilla", 107, "Godzilla_vs._SpaceGodzilla",
          r"space\s?godzilla", ["spacegodzilla"]),
    _film(22, "Godzilla vs. Destoroyah", 103, "Godzilla_vs._Destoroyah",
          r"destoroyah", ["destoroyah"]),
    _film(23, "Godzilla 2000", 107, "Godzilla_2000",
          r"godzilla\s+2000", ["godzilla 2000"]),
    _film(24, "Godzilla vs. Megaguirus", 105, "Godzilla_vs._Megaguirus",
          r"megaguirus", ["megaguirus"]),
    _film(25, "Godzilla, Mothra and King Ghidorah: Giant Monsters All-Out Attack", 105,
          "Godzilla,_Mothra_and_King_Ghidorah:_Giant_Monsters_All-Out_Attack",
          r"giant\s+monsters\s+all[-\s]out\s+attack", ["all-out attack"]),
    _film(26, "Godzilla Against Mechagodzilla", 88, "Godzilla_Against_Mechagodzilla",
          r"godzilla\s+against\s+mechagodzilla", ["against mechagodzilla"]),
    _film(27, "Godzilla: Tokyo S.O.S.", 91, "Godzilla:_Tokyo_S.O.S.",
          r"tokyo\s+s\.?\s?o\.?\s?s", ["tokyo s.o.s"]),
    _film(28, "Godzilla: Final Wars", 125, "Godzilla:_Final_Wars",
          r"final\s+wars", ["final wars"]),
    _film(29, "Shin Godzilla", 120, "Shin_Godzilla",
          r"shin\s+godzilla", ["shin godzilla"]),
    _film(30, "Godzilla: Planet of the Monsters", 88, "Godzilla:_Planet_of_the_Monsters",
          r"planet\s+of\s+the\s+monsters", ["planet of the monsters"]),
    _film(31, "Godzilla: City on the Edge of Battle", 100, "Godzilla:_City_on_the_Edge_of_Battle",
          r"city\s+on\s+the\s+edge", ["city on the edge"]),
    _film(32, "Godzilla: The Planet Eater", 91, "Godzilla:_The_Planet_Eater",
          r"planet\s+eater", ["planet eater"]),
]

# TRUE superlatives only: "shorter than X", "a short film" and "runs 70 minutes" must not open
# the keystone gate on their own.
_SUPERLATIVE = (
    r"shortest|briefest"
    r"|(?:least|lowest|smallest|minimum)\s+(?:running\s+time|runtime|run\s+time|length)"
    r"|(?:running\s+time|runtime)\s+is\s+the\s+(?:lowest|smallest)"
)


def roster(n: int) -> List[Dict[str, Any]]:
    """Return the first ``n`` films of the ordered roster.

    :param n: roster size; must be one of :data:`SWEEP_SIZES`.
    :returns: list of roster-entry dicts, in roster order.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    if n not in SWEEP_SIZES:
        raise ValueError(f"unsupported sweep size {n!r}; expected one of {SWEEP_SIZES}")
    return FILMS[:n]


def keystone(n: int) -> Dict[str, Any]:
    """Return the argmin (shortest-running-time) film of the ``n``-film roster.

    :param n: roster size.
    :returns: the roster-entry dict with the smallest runtime.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    return min(roster(n), key=lambda f: f["runtime"])


def margin(n: int) -> int:
    """Return the argmin margin: runner-up runtime minus the shortest runtime, in minutes.

    :param n: roster size.
    :returns: positive integer margin (7, 5, 11, 11 for N=4, 8, 16, 32).
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    runtimes = sorted(f["runtime"] for f in roster(n))
    return runtimes[1] - runtimes[0]


def total_runtime(n: int) -> int:
    """Return the combined running time of the ``n``-film roster, in minutes.

    :param n: roster size.
    :returns: 362 / 722 / 1405 / 3051 for N=4 / 8 / 16 / 32.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    return sum(f["runtime"] for f in roster(n))


def segments(n: int) -> List[List[int]]:
    """Split the roster indices into three near-equal contiguous position bands.

    The bands are the positional axis of the experiment: a fixed sliding scratchpad evicts the
    EARLY band first, so early-band recall is the direct read-out of the forgetting mechanism.

    :param n: roster size.
    :returns: ``[early, middle, late]``, each a list of 1-based roster indices.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    indices = [f["index"] for f in roster(n)]
    base, extra = divmod(n, 3)
    sizes = [base + 1] * extra + [base] * (3 - extra)
    out, cursor = [], 0
    for size in sizes:
        out.append(indices[cursor:cursor + size])
        cursor += size
    return out


def resolved_indices(text: str, n: int, window: int = _PAIR_WINDOW) -> List[int]:
    """Return the roster indices whose title AND correct runtime are PAIRED in the answer.

    Pairing rule: every occurrence of a runtime figure is attributed to the roster title
    NEAREST to it in the text, and an item counts as resolved only when some occurrence of its
    own runtime is attributed to its own title, within ``window`` characters. A plain "both
    appear somewhere" test would be far too generous -- 12 of the 32 runtimes are shared by at
    least one other film -- and a plain proximity window is too generous as well: with one row
    per film, an answer whose figures are all shifted by one row still puts each film's correct
    figure a dozen characters from its title, and nearest-title attribution is what rejects it.

    :param text: the agent's final answer text.
    :param n: roster size.
    :param window: maximum character gap allowed between a title and the figure attributed to it.
    :returns: sorted list of 1-based indices resolved with the correct value.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    low = text.lower()
    films = roster(n)
    title_spans = [(m.start(), m.end(), f["index"])
                   for f in films for m in re.finditer(f["title_rx"], low)]
    if not title_spans:
        return []
    out = set()
    for film in films:
        for value in re.finditer(film["value_rx"], low):
            before = [(value.start() - end, index) for start, end, index in title_spans
                      if end <= value.start()]
            after = [(start - value.end(), index) for start, end, index in title_spans
                     if start >= value.end()]
            owner = min(before) if before and min(before)[0] <= window else (
                min(after) if after else None)
            if owner is not None and owner[0] <= window and owner[1] == film["index"]:
                out.add(film["index"])
                break
    return sorted(out)


def named_indices(text: str, n: int) -> List[int]:
    """Return the roster indices whose title appears at all, regardless of its runtime.

    Separating "named the film" from "resolved its runtime" tells a roster-discovery failure
    apart from a value-retention failure -- the distinction this whole sweep exists to measure.

    :param text: the agent's final answer text.
    :param n: roster size.
    :returns: sorted list of 1-based indices named anywhere in the answer.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    low = text.lower()
    return [f["index"] for f in roster(n) if re.search(f["title_rx"], low)]


def _visit_count(observability: Dict[str, Any]) -> int:
    """Return the page-visit count from an observability dict, defaulting to 0.

    :param observability: the observability dict (may be ``None`` or missing keys).
    :returns: non-negative visit count.
    """
    return int((observability or {}).get("visit", {}).get("count", 0) or 0)


def keystone_ok(result: Dict[str, Any], observability: Dict[str, Any], n: int) -> bool:
    """Return whether the GROUNDED keystone (argmin film + its runtime) is present.

    Grounding (``visit.count > 0``) is part of the gate so a parametric guess earns nothing.

    :param result: the agent result dict.
    :param observability: the observability dict.
    :param n: roster size.
    :returns: True only when at least one page was visited AND the answer names the shortest
        film as the shortest AND reports its exact runtime.
    """
    if _visit_count(observability) <= 0:
        return False
    text = extract_final_text(result)
    target = keystone(n)
    near = re.compile(
        rf"(?:{_SUPERLATIVE})[^.]{{0,90}}{target['title_rx']}"
        rf"|{target['title_rx']}[^.]{{0,110}}(?:{_SUPERLATIVE})",
        re.IGNORECASE,
    )
    return bool(near.search(text) and re.search(target["value_rx"], text))


def get_task_statement(n: int) -> str:
    """Return the mandate for roster size ``n``.

    Wording is identical at every N apart from the number itself. No film title, runtime,
    total or answer appears -- the roster is earned by reading the named list page.

    :param n: roster size.
    :returns: the task statement string handed to the agent.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    roster(n)
    return (
        "You are given ONE source page and NO list of films.\n\n"
        f"STEP 1. Open the English Wikipedia article '{ROSTER_PAGE}'. Its tables number the "
        "Japanese (Toho) Godzilla films in release order in a numbered column. Read off the "
        f"first {n} of them - numbers 1 to {n} - in that order. Those {n} films are your "
        "roster. Use only the Japanese (Toho) films; ignore the American films listed "
        "separately on the same page.\n"
        f"STEP 2. For EACH film in the roster (the {n} lookups are independent - order does "
        "not matter): open that film's own Wikipedia article and read its RUNNING TIME in "
        "minutes from the infobox. Use the running time in the film's own infobox, which is "
        "the original Japanese release; do NOT use the running time of a re-edited American "
        "version discussed in the article body.\n"
        f"STEP 3. AGGREGATE across the whole roster of {n}: which film has the SHORTEST "
        "running time?\n\n"
        "Report (a) the film with the shortest running time, stating explicitly that it is "
        "the shortest and giving its running time in minutes; (b) the full roster as number - "
        f"title - running time, one row per film, all {n} rows; (c) the combined total running "
        f"time of all {n} films in minutes; and (d) the exact source URL of every film page "
        "you read."
    )


def get_required_deliverables(n: int) -> List[str]:
    """Return the deliverables a complete answer must contain.

    :param n: roster size.
    :returns: list of deliverable descriptions.
    """
    return [
        "The film with the shortest running time, with that running time in minutes",
        f"All {n} roster rows as number - title - running time",
        f"The combined total running time of the {n} films",
        "Source URL per film page read",
    ]


def get_success_criteria(n: int) -> List[str]:
    """Return the human-readable success criteria.

    :param n: roster size.
    :returns: list of criteria strings.
    """
    target = keystone(n)
    return [
        f"Discovers the roster: the first {n} Toho films from '{ROSTER_PAGE}' (roster is not given)",
        f"Reports a running time for every one of the {n} films",
        f"Correctly names the shortest film ({target['title']}, {target['runtime']} minutes)",
        f"Reports the combined total running time ({total_runtime(n)} minutes)",
        "Cites the film pages it read",
    ]


def _make_visits(n: int) -> Callable:
    """Build the un-gated effort diagnostic for roster size ``n``.

    :param n: roster size.
    :returns: validator callable scoring visits against the ``1 + n`` ideal.
    """
    def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
        """Un-gated effort diagnostic: page visits against the 1 roster + n film-page ideal.

        :param result: the agent result dict (unused).
        :param observability: the observability dict.
        :returns: validator dict; ``score`` is visits/(n+1) capped at 1.0, ``passed`` at >= 2.
        """
        count = _visit_count(observability)
        target = n + 1
        return {"check": "visit_count", "passed": count >= 2,
                "score": min(1.0, count / float(target)),
                "reason": f"{count} visit(s) (ideal >={target}: 1 roster page + {n} film pages; "
                          f">=2 to pass)"}
    return validate_visits


def _make_keystone(n: int) -> Callable:
    """Build the hard 0/1 keystone validator for roster size ``n``.

    :param n: roster size.
    :returns: validator callable scoring 1.0 or 0.0.
    """
    target = keystone(n)

    def validate_keystone_shortest(result: Dict[str, Any],
                                   observability: Dict[str, Any]) -> Dict[str, Any]:
        """KEYSTONE (hard 0/1): the argmin film named as the shortest, with its exact runtime,
        and at least one page actually visited.

        :param result: the agent result dict.
        :param observability: the observability dict.
        :returns: validator dict scoring 1.0 or 0.0.
        """
        passed = keystone_ok(result, observability, n)
        return {"check": "keystone_shortest_film", "passed": passed,
                "score": 1.0 if passed else 0.0,
                "reason": (f"Shortest of the first {n} = {target['title']} "
                           f"({target['runtime']} min)") if passed else
                          (f"Keystone missing/incorrect (expected {target['title']}, "
                           f"{target['runtime']} minutes, with >=1 page visited)")}
    return validate_keystone_shortest


def _make_coverage(n: int) -> Callable:
    """Build the un-gated per-item recall validator for roster size ``n``.

    :param n: roster size.
    :returns: validator callable scoring resolved items / n.
    """
    def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
        """HEADLINE DIAGNOSTIC (UN-gated): exact per-item recall, k/n, an item counting only
        when its title and its correct runtime are paired in the answer.

        Never short-circuited on the keystone: it measures how much of the roster survived even
        when the final argmin is botched, which is the axis that separates a coverage ledger
        from a sliding scratchpad.

        :param result: the agent result dict.
        :param observability: the observability dict (unused).
        :returns: validator dict; ``score`` is resolved/n, with the resolved and named index
            lists in ``detail`` for post-hoc positional analysis.
        """
        text = extract_final_text(result)
        resolved = resolved_indices(text, n)
        named = named_indices(text, n)
        return {"check": "coverage", "passed": len(resolved) == n,
                "score": len(resolved) / float(n),
                "reason": f"{len(resolved)}/{n} items resolved with the correct runtime "
                          f"(indices {resolved or 'none'}); {len(named)}/{n} titles named at all",
                "detail": {"n": n, "resolved": resolved, "named": named}}
    return validate_coverage


def _make_positional(n: int) -> Callable:
    """Build the un-gated positional-recall validator for roster size ``n``.

    :param n: roster size.
    :returns: validator callable scoring EARLY-band recall.
    """
    bands = segments(n)

    def validate_positional_recall(result: Dict[str, Any],
                                   observability: Dict[str, Any]) -> Dict[str, Any]:
        """MECHANISM DIAGNOSTIC (UN-gated): recall split by position in the roster.

        Scored on the EARLY band because that is the mechanism prediction under test: a fixed
        12-step sliding scratchpad evicts the items it gathered FIRST, so early-band recall
        should collapse with N while late-band recall holds. ``reason`` and ``detail`` carry
        all three bands so the middle/late comparison is available without rescoring.

        :param result: the agent result dict.
        :param observability: the observability dict (unused).
        :returns: validator dict; ``score`` is early-band resolved / early-band size.
        """
        resolved = set(resolved_indices(extract_final_text(result), n))
        hits = [[i for i in band if i in resolved] for band in bands]
        labels = ("early", "middle", "late")
        parts = [f"{labels[k]} {len(hits[k])}/{len(bands[k])}" for k in range(3)]
        early_score = len(hits[0]) / float(len(bands[0]))
        return {"check": "positional_recall", "passed": early_score == 1.0,
                "score": early_score,
                "reason": f"positional recall - {', '.join(parts)} "
                          f"(scored on the early band, indices {bands[0]})",
                "detail": {"bands": bands, "hits": hits,
                           "band_scores": [len(hits[k]) / float(len(bands[k])) for k in range(3)]}}
    return validate_positional_recall


def _make_total(n: int) -> Callable:
    """Build the gated total-running-time validator for roster size ``n``.

    :param n: roster size.
    :returns: validator callable scoring 1.0 or 0.0.
    """
    total = total_runtime(n)
    hours, minutes = divmod(total, 60)
    pattern = re.compile(
        rf"\b{total // 1000},?{total % 1000:03d}\b" if total >= 1000 else rf"\b{total}\b"
    )
    hm_pattern = re.compile(rf"\b{hours}\s*(?:h|hr|hrs|hours?)\b[^.]{{0,20}}\b{minutes}\b",
                            re.IGNORECASE)

    def validate_total_runtime(result: Dict[str, Any],
                               observability: Dict[str, Any]) -> Dict[str, Any]:
        """Gated secondary: the combined running time of the whole roster, which is wrong
        unless every one of the n items was gathered. Short-circuits to 0 without the keystone.

        :param result: the agent result dict.
        :param observability: the observability dict.
        :returns: validator dict scoring 1.0 or 0.0.
        """
        if not keystone_ok(result, observability, n):
            return {"check": "total_runtime", "passed": False, "score": 0.0,
                    "reason": "Keystone absent -> total running time not credited"}
        text = extract_final_text(result)
        hit = bool(pattern.search(text) or hm_pattern.search(text))
        return {"check": "total_runtime", "passed": hit, "score": 1.0 if hit else 0.0,
                "reason": f"combined running time {total} min {'reported' if hit else 'missing'}"}
    return validate_total_runtime


def _make_citations(n: int) -> Callable:
    """Build the gated citation validator for roster size ``n``.

    :param n: roster size.
    :returns: validator callable scoring cited film pages / n.
    """
    def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
        """Gated secondary: per-film source URLs. Short-circuits to 0 without the keystone.

        :param result: the agent result dict.
        :param observability: the observability dict.
        :returns: validator dict; ``score`` is cited films / n, ``passed`` at >= half.
        """
        if not keystone_ok(result, observability, n):
            return {"check": "citations", "passed": False, "score": 0.0,
                    "reason": "Keystone absent -> source URLs not credited"}
        text = extract_final_text(result).lower()
        cited = [f["index"] for f in roster(n) if re.search(f["slug_rx"], text)]
        return {"check": "citations", "passed": len(cited) * 2 >= n,
                "score": len(cited) / float(n),
                "reason": f"{len(cited)}/{n} film pages cited"}
    return validate_citations


def get_validation_functions(n: int) -> List[Callable]:
    """Return the validators for roster size ``n``, keystone first among the scored checks.

    Two un-gated diagnostics (coverage, positional recall) and three keystone-gated checks
    (visits is un-gated but is an effort meter, not an answer credit), so a run that gathers
    everything and then botches the argmin lands near 0.5 instead of 1.0, and a run that
    gathers nothing lands near 0.

    :param n: roster size.
    :returns: list of validator callables.
    """
    return [_make_visits(n), _make_keystone(n), _make_coverage(n), _make_positional(n),
            _make_total(n), _make_citations(n)]


def get_compiled_plan(n: int) -> Dict[str, Any]:
    """Return the offline-authored DAG (schema v2) for the ``graph_compiled`` arm.

    One roster leaf, then ``n`` independent film leaves that address their subject POSITIONALLY
    ("film number i of that roster") and receive it through ``{roster}`` templating. Leaks
    nothing: the only proper noun is the GIVEN roster page; no film title, no runtime, no total
    and no argmin appear anywhere in the plan.

    :param n: roster size.
    :returns: plan dict with ``leaves`` and ``aggregation``.
    :raises ValueError: if ``n`` is not a supported sweep size.
    """
    roster(n)
    leaves: List[Dict[str, Any]] = [{
        "id": "roster",
        "instruction": (
            f"Open the English Wikipedia article '{ROSTER_PAGE}'. Its tables number the "
            "Japanese (Toho) films in release order in a numbered column. List the first "
            f"{n} of them - numbers 1 to {n} - one film per line, in that order. Use only "
            "the Japanese (Toho) films; ignore the American films listed separately."
        ),
        "expect": f"NUMBERED LIST of {n} film titles - the list page's exact Wikipedia URL",
        "depends_on": [],
    }]
    for i in range(1, n + 1):
        leaves.append({
            "id": f"film_{i}",
            "instruction": (
                f"Here is a numbered list of films: {{roster}}\nTake film number {i} of that "
                "list. Open that film's own Wikipedia article and read its RUNNING TIME in "
                "minutes from the infobox. Use the infobox figure for the original Japanese "
                "release, not the running time of a re-edited American version discussed in "
                "the article body."
            ),
            "expect": "FILM TITLE - RUNNING TIME in minutes - the film page's exact Wikipedia URL",
            "depends_on": ["roster"],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            f"You are given, for each of the {n} films on the discovered roster, its running "
            "time in minutes with a source URL. AGGREGATE across the whole roster: find the "
            "MINIMUM running time. Report (a) the film with the shortest running time, stating "
            "explicitly that it is the shortest and giving its running time in minutes; (b) "
            f"all {n} rows as number - title - running time; (c) the combined total running "
            f"time of the {n} films in minutes; and (d) the source URL of every film page."
        ),
    }
