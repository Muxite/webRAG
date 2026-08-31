#!/usr/bin/env python3
"""
validator_lint.py: static integrity linter for webRAG idea_tests task validators.

Promoted from the pre-barrage audit scratchpad (.barrage_prep/validator_lint.py, AGENT5) into a
tracked, CI-wired check (F30). Detection logic is unchanged from the audited version.

Flags, per task file:
  [GATE]  keystone/answer validators that can score >0 with ZERO grounding
          (no observability visit gate AND no build_visit_link_graph gate).
  [LLM]   a non-None LLM judge (violates the deterministic-only validity bar).
  [LEAK]  a compiled-plan ``verify`` leaf whose ``details.optional_url``/``details.url`` is set
          to a module-level authoritative-URL constant. ``VerifyLeafAction`` (agent/app/
          idea_policies/actions.py) auto-fetches that field and inserts the page at
          evidence[0] before its LLM call, grounding the reconcile step off the answer page
          directly regardless of what upstream visit leaves actually found -- the same leak
          found+fixed across six tasks on 2026-08-31 (commit 4bd17b0a). [LEAK] makes that bug
          class un-reintroducible.
  [UNIT]  abbreviation-only unit keystones ( \\d\\s*m\\b / \\s*mph / \\s*kn )
          with NO bare-number or spelled-unit fallback -> "300 metres" false-fails.
  [DEC]   no-tolerance decimal keystones (\\bNN\\.NN\\b) -> standard roundings false-fail.

CLI usage:  python scripts/validator_lint.py [dir]   (default: agent/app/idea_tests)
Exit 1 if any [GATE], [LLM], or [LEAK] finding (the three score-corrupting severities).

Programmatic usage (e.g. from a pytest CI gate):
    from validator_lint import lint_directory
    findings = lint_directory(some_dir)   # -> List[Tuple[task_id: str, finding: str]]
"""
import ast
import glob
import os
import re
import sys
from typing import List, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(_REPO_ROOT, "agent", "app", "idea_tests")


def _fn(tree: ast.AST, name: str):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def _seg(node, src: str) -> str:
    return ast.get_source_segment(src, node) or "" if node is not None else ""


# keystone regex literals: KEYSTONE_RX / _CORRECT_RE / *_RX = re.compile(r"...")
_KS_ASSIGN = re.compile(r"(?:KEYSTONE\w*|_CORRECT_RE)\s*=\s*re\.compile\(\s*r?([\"'])(.*?)\1", re.S)
_ABBREV_UNIT = re.compile(r"\\s\*(?:m|mph|kn|km|ft)\\b")
_BARE_NUM_ALT = re.compile(r"\\b\d[,\\s\?\d]*\\b")          # a bare-number alternative branch
_DECIMAL_KS = re.compile(r"\\b\d+\\\.\d+\\b")

# only the discriminating "answer" validators matter (keystone / survivor / value / citation)
_ANSWER_LIKE = re.compile(
    r"keystone|survivor|winner|value|citation|argmax|chain|height|"
    r"length|depth|density|radius|scale|temp|carat|runway|elevation|"
    r"population|count|steps|drop|antenna|banyan|difference",
    re.I,
)


def _dict_get(node, key: str):
    """Return the value node for ``key`` in an ``ast.Dict`` literal, or ``None``."""
    if not isinstance(node, ast.Dict):
        return None
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _find_verify_url_leaks(tree: ast.AST) -> List[str]:
    """Find compiled-plan ``verify`` leaves whose ``details.optional_url``/``details.url`` is
    wired to a module-level authoritative-URL constant (see the [LEAK] docstring above)."""
    # Collect module-level "authoritative URL" constants: NAME containing both AUTHORITATIVE
    # and URL, assigned a (possibly implicitly-concatenated) string literal.
    auth_values = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if re.search(r"AUTHORITATIVE", name, re.I) and re.search(r"URL", name, re.I):
                try:
                    val = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    val = None
                if isinstance(val, str):
                    auth_values.add(val)

    findings: List[str] = []
    if not auth_values:
        return findings

    for n in ast.walk(tree):
        if not isinstance(n, ast.Dict):
            continue
        action_v = _dict_get(n, "action")
        if not (isinstance(action_v, ast.Constant) and action_v.value == "verify"):
            continue
        details_v = _dict_get(n, "details")
        if not isinstance(details_v, ast.Dict):
            continue
        for key_name in ("optional_url", "url"):
            val_node = _dict_get(details_v, key_name)
            if val_node is None:
                continue
            leaked_value = None
            try:
                literal = ast.literal_eval(val_node)
            except (ValueError, TypeError):
                literal = None
            if isinstance(literal, str) and literal in auth_values:
                leaked_value = literal
            elif isinstance(val_node, ast.Name) and re.search(r"AUTHORITATIVE", val_node.id, re.I):
                leaked_value = val_node.id
            if leaked_value is not None:
                node_id_v = _dict_get(n, "id")
                nid = node_id_v.value if isinstance(node_id_v, ast.Constant) else "?"
                findings.append(
                    f"[LEAK] verify leaf {nid!r}: details.{key_name} set to the authoritative "
                    f"URL constant -- VerifyLeafAction will auto-fetch it and ground the "
                    f"reconcile step off the answer page regardless of upstream visits"
                )
    return findings


def lint_file(path: str) -> List[str]:
    """
    Lint a single idea_tests task file.
    :param path: Absolute path to a ``test_NNN_*.py`` task file.
    :return: List of finding strings, each prefixed with its severity tag
             (``[GATE]``, ``[LLM]``, ``[LEAK]``, ``[UNIT]``, ``[DEC]``).
    """
    src = open(path).read()
    tree = ast.parse(src)
    findings: List[str] = []

    # --- [LLM] ---
    llm = _fn(tree, "get_llm_validation_function")
    if llm and "return None" not in _seg(llm, src):
        findings.append("[LLM]  get_llm_validation_function returns a judge")

    # --- [LEAK] ---
    findings.extend(_find_verify_url_leaks(tree))

    # --- [GATE] ---
    uses_graph = bool(re.search(
        r"build_visit_link_graph|_hop_visited|visit_adjacency_map|visited_url_set", src,
    ))
    gvf = _fn(tree, "get_validation_functions")
    val_names = re.findall(r"\b(validate_\w+)", _seg(gvf, src)) if gvf else []
    for vn in val_names:
        vf = _fn(tree, vn)
        if not vf:
            continue
        vs = _seg(vf, src)
        # a validator is grounding-independent if its body (and any _keystone_ok it calls,
        # when that keystone doesn't itself ground) never consults visits or the visit-graph.
        # `waypoint_chain_coverage`/`waypoint_evidence_ok`/`visited_evidence` (idea_test_utils.py)
        # are the shared per-waypoint grounding helpers the chain_coverage repair (2026-08-16)
        # factored the visit-evidence check into -- a validator that delegates to one of them is
        # grounded even though the grounding logic itself now lives outside this file's AST.
        # `visit_adjacency_map`/`visited_url_set` (2026-08-31) are the arm-symmetric replacements
        # for `build_visit_link_graph` (which reads exclusively from result["graph"], populated
        # only by the graph/naive_discretion arms) -- also grounding helpers.
        grounds = bool(re.search(
            r"observability.*visit|\[.visit.\]|_hop_visited|build_visit_link_graph|"
            r"waypoint_chain_coverage|waypoint_evidence_ok|visited_evidence|"
            r"visit_adjacency_map|visited_url_set",
            vs,
        ))
        calls_ks = "_keystone_ok(" in vs
        ks = _fn(tree, "_keystone_ok")
        ks_grounds = bool(ks and re.search(r"visit|_hop_visited|build_visit", _seg(ks, src)))
        answer_like = _ANSWER_LIKE.search(vn)
        if answer_like and not grounds and not (calls_ks and ks_grounds) and not uses_graph:
            findings.append(f"[GATE] {vn} scores without grounding")

    # --- [UNIT] / [DEC] on keystone regexes ---
    for _q, pat in _KS_ASSIGN.findall(src):
        branches = pat.split("|")
        # abbreviation-only unit with no bare-number and no spelled-unit alternative
        if _ABBREV_UNIT.search(pat) and not _BARE_NUM_ALT.search(pat) \
           and "metre" not in pat and "meter" not in pat and "mile" not in pat:
            findings.append(f"[UNIT] keystone {pat!r} abbrev-only unit; 'NNN metres' will false-fail")
        for b in branches:
            if _DECIMAL_KS.search(b) and "[" not in b:   # a fixed decimal, no tolerance/char-class
                findings.append(f"[DEC]  keystone branch {b.strip()!r} no rounding tolerance")

    return findings


def lint_directory(directory: str = DEFAULT_DIR) -> List[Tuple[str, str]]:
    """
    Lint every ``test_*.py`` file in ``directory``.
    :param directory: Directory containing idea_tests task files.
    :return: List of ``(task_id, finding)`` tuples, e.g. ``("024", "[LLM]  ...")``.
    """
    results: List[Tuple[str, str]] = []
    for f in sorted(glob.glob(os.path.join(directory, "test_*.py"))):
        base = os.path.basename(f)
        tid = base[5:8]
        for finding in lint_file(f):
            results.append((tid, finding))
    return results


def hard_findings(findings: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter to the three score-corrupting severities: [GATE], [LLM], and [LEAK]."""
    return [(tid, fi) for tid, fi in findings if fi.startswith(("[GATE]", "[LLM]", "[LEAK]"))]


def main(argv: List[str]) -> int:
    directory = argv[1] if len(argv) > 1 else DEFAULT_DIR
    findings = lint_directory(directory)
    for tid, fi in findings:
        print(f"{tid}  {fi}")
    hard = hard_findings(findings)
    print(f"\n{len(hard)} score-corrupting [GATE]/[LLM]/[LEAK] findings")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
