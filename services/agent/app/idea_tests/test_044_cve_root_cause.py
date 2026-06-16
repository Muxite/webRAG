"""
Test 044: Security Vulnerability Root-Cause Analysis
Difficulty: 10/10 (Maximum)
Category: Security Vulnerability Root-Cause Analysis

The agent must combine a CVE advisory with the actual OpenSSH source code to pin down the
source-level root cause. The keystone (the name of the vulnerable C function) does NOT
appear in the NVD/vendor summary pages or even the high-level write-ups — it is only in the
source. So the agent must (1) read the C source file, (2) identify which function performs
the certificate-principal matching the CVE describes, and (3) cross-reference the advisory
for the fix version and affected config. This rewards reading code, not just summaries:

- `parametric` (training cutoff before the disclosure) cannot know an April-2026 CVE -> ~0.
- A model that only reads the summary advisory names the option/version but NOT the function
  -> misses the keystone (caps at the visit floor).
- A model that visits the seeded source file AND identifies the right function by its role
  -> wins.

Two authoritative URLs are named in the mandate (so the task does not depend on a flaky
search and is reproducible under fixtures): the OpenSSH source file and a CVE advisory.

Scoring is gated/bimodal: the keystone is the vulnerable function name; secondary checks
short-circuit to 0.0 when it is absent.

Ground truth (CVE-2026-35414, OpenSSH; verified findable via live web search 2026-06):
  vulnerable function : match_principals_option
  source file         : auth2-pubkeyfile.c
  unsafe helper used  : match_list  (treats a comma inside one principal as a separator)
  correct comparison  : strcmp      (used by the fix and by the safe principals-file path)
  fixed in            : OpenSSH 10.3 / 10.3p1
  affected config     : cert-authority + principals= in authorized_keys
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CVE_ID = "CVE-2026-35414"
KEYSTONE_FUNCTION = r"match_principals_option"
SOURCE_URL = "https://raw.githubusercontent.com/openssh/openssh-portable/master/auth2-pubkeyfile.c"
ADVISORY_URL = "https://ubuntu.com/security/CVE-2026-35414"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "044",
        "test_name": "Security Vulnerability Root-Cause Analysis",
        "difficulty_level": "10/10",
        "category": "Security Vulnerability Root-Cause Analysis",
        "level": "integration",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Investigate the OpenSSH vulnerability {CVE_ID} (a principal-escalation bug in sshd "
        "certificate authentication: a comma inside a certificate principal is mishandled when "
        "matching the authorized_keys principals= option).\n\n"
        "You MUST consult BOTH of these pages:\n"
        f"  - OpenSSH source file: {SOURCE_URL}\n"
        f"  - CVE advisory: {ADVISORY_URL}\n\n"
        "Read the C source and identify the single function that matches a certificate's principals "
        "against the authorized_keys principals= option (the function the CVE concerns). Then report "
        "ALL of the following:\n"
        "  1. The exact name of that vulnerable C function.\n"
        "  2. The C source file it lives in.\n"
        "  3. The string-comparison function used to match a principal (what the fix uses).\n"
        "  4. The OpenSSH version that fixed the bug.\n"
        "  5. The authorized_keys trust configuration that is affected (the key option).\n"
        "Cite the source URLs you used. Do not answer from memory — verify against the pages you visit."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Vulnerable C function name",
        "Source file name",
        "String-comparison function used by the fix (strcmp)",
        "Fixed OpenSSH version",
        "Affected authorized_keys configuration",
        "Source URL citation(s)",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 visit actions executed (the source file and the advisory)",
        "Identifies the vulnerable function match_principals_option",
        "Identifies the comparison function strcmp used by the fix",
        "Identifies source file auth2-pubkeyfile.c and fix version 10.3",
        "Identifies the cert-authority/principals= trust path",
        "Cites source URLs",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_FUNCTION, extract_final_text(result), re.IGNORECASE))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    return {
        "check": "visit_count",
        "passed": visit_count >= 2,
        "score": min(1.0, visit_count / 3.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=2)",
    }


def validate_keystone_function(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the exact vulnerable function name. Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_vulnerable_function",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Identified match_principals_option" if passed else "Vulnerable function name missing",
    }


def validate_root_cause(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """The comparison function (strcmp); match_list is a bonus. Short-circuits when keystone absent.

    The seeded source is the *fixed* file (uses strcmp); match_list only appears in the
    vulnerable version / diff, so it is credited as a bonus but not required.
    """
    if not _keystone_ok(result):
        return {"check": "root_cause", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> root cause not credited"}
    text = extract_final_text(result)
    has_strcmp = bool(re.search(r"strcmp", text, re.IGNORECASE))
    has_match_list = bool(re.search(r"match_list", text, re.IGNORECASE))
    score = 1.0 if has_strcmp else (0.5 if has_match_list else 0.0)
    return {
        "check": "root_cause",
        "passed": has_strcmp or has_match_list,
        "score": score,
        "reason": f"strcmp={has_strcmp}, match_list(bonus)={has_match_list}",
    }


def validate_file_and_fix(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Source file, fix version, and affected config. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "file_fix_config", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> file/fix/config not credited"}
    text = extract_final_text(result)
    low = text.lower()
    has_file = bool(re.search(r"auth2-pubkeyfile\.c", low))
    has_fix = bool(re.search(r"10\.3(p1)?", low))
    has_config = ("cert-authority" in low) or bool(re.search(r"principals\s*=", low))
    hits = int(has_file) + int(has_fix) + int(has_config)
    return {
        "check": "file_fix_config",
        "passed": hits >= 2,
        "score": hits / 3.0,
        "reason": f"file={has_file}, fix_10.3={has_fix}, cert_config={has_config}",
    }


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Source URLs cited. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
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
    return [validate_visits, validate_keystone_function, validate_root_cause,
            validate_file_and_fix, validate_citations]


def get_llm_validation_function() -> callable:
    return None
