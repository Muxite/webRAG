"""
Test 093: Tier 5 (cross-domain heterogeneous chain) — CVE advisory -> C source root-cause (B)
Level: graph   Weight: long   Difficulty: 10/10

A revival of the original-suite heterogeneous-chain idea (tests 028/031) applied fresh, and a
deliberate SOURCE-SHAPE counterpoint to the wiki-infobox tasks that dominate tiers 040-089:
the chain runs advisory (a summary web page) -> actual C source code. It is the sibling of
test 044 (OpenSSH CVE root-cause) but on a DIFFERENT vulnerability, source tree and keystone.

The task: pin down the ROOT CAUSE of the curl SOCKS5 heap-overflow CVE-2023-38545 at the
source level. The official curl advisory and the NVD entry both describe the bug by PARAPHRASE
— "the local variable that means 'let the host resolve the name' could get the wrong value
during a slow SOCKS5 handshake" — but NEITHER names the variable, the function, or the file.
So the keystone (the exact C identifier of that flag) is recoverable ONLY by reading the seeded
C source. This rewards reading code, not summaries:

- `parametric` / summary-only agents that read just the advisory get the *concept* but never the
  exact identifier -> miss the keystone.
- An agent that opens the seeded lib/socks.c and finds the flag variable by its role -> wins.

Following the 044 lesson (a pure "go research the CVE" version stalls after one summary page),
TWO authoritative URLs are seeded directly in the mandate so the task is reproducible under
fixtures and does not depend on a flaky search: the vulnerable curl source file (pinned to the
curl-8_3_0 tag) and the official curl advisory.

Scoring is gated/bimodal: the keystone is the exact flag variable name; the citation secondary
short-circuits to 0.0 when it is absent; the coverage diagnostic is UN-gated (it measures how
much of the cross-domain reading actually happened even when the exact identifier is botched).

Ground truth (CVE-2023-38545, curl/libcurl; verified against live sources 2026-07-07):
  keystone flag variable : socks5_resolve_local   (lib/socks.c line 573, curl-8_3_0)
  vulnerable function    : do_SOCKS5               (lib/socks.c line 548; renamed Curl_SOCKS5 later)
  source file            : socks.c  (lib/socks.c)
  length threshold       : 255      (RFC1928 max SOCKS5 domain-name length; line 589)
  overflow primitive     : memcpy   (heap buffer overflow; line 907)
  fixed in               : curl 8.4.0   (affected 7.69.0 .. 8.3.0 inclusive)

Leak-resistance verified 2026-07-07: the exact identifier `socks5_resolve_local` (and the
function `do_SOCKS5` and file `socks.c`) appear in NONE of the summary sources checked — the
NVD entry (nvd.nist.gov/vuln/detail/CVE-2023-38545), the official curl advisory
(curl.se/docs/CVE-2023-38545.html) and the Tenable scanner summary
(tenable.com/cve/CVE-2023-38545) all paraphrase the flag as "the local variable that means
'let the host resolve the name'" without ever naming it. It is present only in the C source.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CVE_ID = "CVE-2023-38545"
SOURCE_URL = "https://raw.githubusercontent.com/curl/curl/curl-8_3_0/lib/socks.c"
ADVISORY_URL = "https://curl.se/docs/CVE-2023-38545.html"

# KEYSTONE: the exact C identifier of the "let the host resolve the name" flag. Source-only.
KEYSTONE_VARIABLE = re.compile(r"socks5_resolve_local", re.IGNORECASE)

# Coverage facts (source + advisory derivable) — the UN-gated breadth axis.
_FUNC = re.compile(r"\b(?:do|curl)_socks5\b", re.IGNORECASE)   # do_SOCKS5 (or renamed Curl_SOCKS5)
_FILE = re.compile(r"socks\.c", re.IGNORECASE)
_THRESHOLD = re.compile(r"\b255\b")
_OVERFLOW = re.compile(r"memcpy|heap[^.]{0,24}overflow|buffer\s+overflow", re.IGNORECASE)
_FIX = re.compile(r"\b8\.4\.0\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "093",
        "test_name": "Tier 5: CVE Advisory -> C Source Root-Cause Chain (B)",
        "difficulty_level": "10/10",
        "category": "Security CVE Root-Cause / Cross-Domain Chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        f"Investigate the curl/libcurl vulnerability {CVE_ID} (a heap buffer overflow in the "
        "SOCKS5 proxy handshake). The advisory explains that when curl asks the SOCKS5 proxy to "
        "resolve the host name itself, a boolean local variable meaning 'let the host resolve the "
        "name' can get the WRONG value during a slow handshake, so an over-long host name is "
        "copied into a fixed target buffer. The advisory paraphrases that flag but never names it.\n\n"
        "You MUST consult BOTH of these pages:\n"
        f"  - curl C source file (pinned): {SOURCE_URL}\n"
        f"  - Official curl advisory: {ADVISORY_URL}\n\n"
        "Read the C source and identify the single local flag variable that decides whether the "
        "proxy resolves the host name (the flag the advisory describes). Then report ALL of:\n"
        "  1. The EXACT name of that flag variable, verbatim as written in the code (the keystone).\n"
        "  2. The name of the C function that contains it.\n"
        "  3. The C source file it lives in.\n"
        "  4. The maximum host-name length (in bytes) the code checks before switching modes.\n"
        "  5. The curl version that fixed the bug.\n"
        "Cite the source URLs you used. Do not answer from memory — verify against the pages you visit."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Exact flag variable name (keystone: socks5_resolve_local)",
        "Vulnerable C function name (do_SOCKS5)",
        "Source file name (socks.c)",
        "Host-name length threshold (255 bytes)",
        "Fixed curl version (8.4.0)",
        "Source URL citation(s)",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 visit actions executed (the source file and the advisory)",
        "Identifies the exact flag variable socks5_resolve_local",
        "Identifies the vulnerable function do_SOCKS5 and file socks.c",
        "Identifies the 255-byte host-name threshold",
        "Identifies fix version curl 8.4.0",
        "Cites source URLs",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Also requires GROUNDING: the exact identifier string alone is insufficient — the agent
    must have actually visited at least one page (visit.count > 0), else an ungrounded
    parametric-memory guess would earn credit."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return bool(KEYSTONE_VARIABLE.search(extract_final_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 2,
        "score": min(1.0, n / 3.0),
        "reason": f"{n} visit(s) (target >=2: the seeded source file and the advisory)",
    }


def validate_keystone_variable(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the exact flag variable identifier (socks5_resolve_local). Hard 0/1.

    Source-only: the advisory/NVD/scanner summaries paraphrase this flag but never name it, so
    an agent that read only a summary page (or answered from memory) cannot produce the exact
    identifier — it must read the seeded C source.
    """
    passed = _keystone_ok(result, observability)
    return {
        "check": "keystone_flag_variable",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Identified socks5_resolve_local" if passed
                  else "Exact flag variable (socks5_resolve_local) missing/incorrect",
    }


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated cross-domain coverage diagnostic: how many supporting root-cause facts were
    gathered (function, file, 255-byte threshold, overflow primitive, fix version).

    Deliberately NOT short-circuited on the keystone — it measures whether the agent actually
    read across BOTH the advisory and the C source (the axis that separates a structured agent
    that reaches source from a linear one that stops at the summary), even when the exact
    identifier is botched.
    """
    text = extract_final_text(result)
    facts = {
        "function_do_socks5": bool(_FUNC.search(text)),
        "file_socks_c": bool(_FILE.search(text)),
        "threshold_255": bool(_THRESHOLD.search(text)),
        "overflow_primitive": bool(_OVERFLOW.search(text)),
        "fix_8_4_0": bool(_FIX.search(text)),
    }
    covered = sum(1 for v in facts.values() if v)
    n = len(facts)
    return {
        "check": "coverage",
        "passed": covered == n,
        "score": covered / n,
        "reason": f"{covered}/{n} supporting facts ("
                  + ", ".join(k for k, v in facts.items() if v) + ")"
                  if covered else f"0/{n} supporting facts",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Source URLs + CVE id cited. Short-circuits to 0.0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = extract_final_text(result)
    urls = re.findall(r"https?://[^\s)\\\"]+", text)
    mentions_cve = CVE_ID.lower() in text.lower()
    score = 0.5 * min(1.0, len(urls) / 2.0) + 0.5 * (1.0 if mentions_cve else 0.0)
    return {
        "check": "citations",
        "passed": len(urls) >= 1 and mentions_cve,
        "score": round(score, 3),
        "reason": f"urls={len(urls)}, mentions_cve={mentions_cve}",
    }


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_variable, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored cross-domain CHAIN scaffold for the ``graph_compiled`` variant.

    Two self-describing leaves forming a short dependent chain (advisory -> source). The first
    reads the seeded advisory for the flag's plain-English MEANING and the fix version; the
    second ``depends_on`` it, templating that meaning via ``{advisory_context}`` so the source
    read is anchored to the right flag. Encodes only STRUCTURE — it names the given CVE and the
    seeded URLs but leaks NO answer (not the variable name, function name, threshold or fix
    version). Every leaf self-describes what it is answering (required to avoid aggregation
    binding loss); the cheap runtime model still does every page-read and extraction.
    """
    return {
        "leaves": [
            {
                "id": "advisory_context",
                "instruction": (
                    f"Open the official curl advisory for {CVE_ID}: {ADVISORY_URL}. This leaf "
                    "answers: what is the PLAIN-ENGLISH meaning of the boolean flag the advisory "
                    "blames (the one about who resolves the host name), which curl VERSION fixed "
                    "the bug, and what heap operation performs the overflowing copy? Report the "
                    "flag's meaning phrase, the fix version and the copy operation."
                ),
                "expect": "flag meaning phrase + fixed curl version + copy operation + advisory URL",
                "depends_on": [],
            },
            {
                "id": "source_flag_identifier",
                "instruction": (
                    "The seeded curl advisory describes a boolean flag whose meaning is: "
                    "{advisory_context}. This leaf answers: in the seeded C source file "
                    f"{SOURCE_URL}, what is the EXACT local-variable identifier that implements "
                    "that flag (verbatim as written in the code), what FUNCTION contains it, and "
                    "what maximum host-name length in bytes does that function check before "
                    "switching resolve modes? Read the source directly; do not guess from memory."
                ),
                "expect": "exact flag variable identifier + containing function name + byte threshold + source URL",
                "depends_on": ["advisory_context"],
            },
        ],
        "aggregation": (
            "Report (a) the EXACT flag variable identifier from the C source (this is the "
            "keystone), (b) the function and source file it lives in and the host-name byte "
            f"threshold it checks, and (c) the curl version that fixed {CVE_ID}. Cite both the "
            "source-file URL and the advisory URL."
        ),
    }
