"""
codebench task c32 — hard/hidden, string/text algorithms: explicit-FSM format validator.

New category coverage beyond the existing suite's RLE (c02) / balanced-brackets (c03) /
Roman-numeral (c13) string tasks, and distinct from c30 (line diff) and c31 (tokenizer) — this
one is a character-by-character FORMAT VALIDATOR with several independent local rules (dot
placement, hyphen placement, label count, TLD shape) that a single regex can express but that
this task deliberately asks to be built as an explicit finite-state machine instead, so a weak
model can't just paste a memorized email regex from training data and be done — it has to
actually reason about each rule as its own state-transition condition. The test cases below
are deliberately chosen to each probe ONE specific rule/transition in isolation (mirrors this
task's design brief: "verify against valid/invalid inputs probing specific state transitions").

Ground truth for ``is_valid_email`` verified with an independently-written two-pass reference
(split on '@' then validate local-part and domain-part each via their own single-character
walk, structured differently from a true one-pass FSM) run and cross-checked against every
case below before any value here was written down — see idea_code_test_c32_test.py for the
reimplementation that re-derives this independently again. Do not hand-edit these literals
without re-verifying the same way.

## Grammar (simplified — not full RFC 5322)

    LOCAL "@" DOMAIN

  * LOCAL: one or more characters from {letters, digits, '.', '_', '%', '+', '-'}. Must not
    start with '.', must not end with '.', must not contain two consecutive dots ('..').
  * DOMAIN: two or more dot-separated LABELs. Each LABEL: one or more characters from
    {letters, digits, '-'}; must not start or end with '-'; must be non-empty (so a leading,
    trailing, or doubled dot in DOMAIN is invalid, since it produces an empty label). The
    LAST label (the TLD) must additionally consist ONLY of letters (no digits, no hyphens)
    and be at least 2 characters long.
  * The whole string must contain EXACTLY one '@'. "Letters" means the 26 ASCII letters,
    upper- or lower-case.

    "a@b.co"                                   -> True   (minimal valid)
    "john.doe@example.com"                     -> True
    "user_name+tag@sub-domain.example.org"     -> True   (3 domain labels, mid-label hyphen)
    "a.b.c@example.co"                         -> True   (multiple non-consecutive local dots)
    "x99@ab12.io"                              -> True   (digits in local + non-TLD label)
    ""                                          -> False  (empty)
    "noatsign.example.com"                     -> False  (no '@' at all)
    "two@at@signs.com"                         -> False  (more than one '@')
    "@example.com"                             -> False  (empty local part)
    "user@"                                     -> False  (empty domain)
    ".user@example.com"                        -> False  (local starts with '.')
    "user.@example.com"                        -> False  (local ends with '.')
    "us..er@example.com"                       -> False  (consecutive dots in local)
    "user@.example.com"                        -> False  (leading dot in domain -> empty label)
    "user@example.com."                        -> False  (trailing dot in domain -> empty label)
    "user@example..com"                        -> False  (consecutive dots in domain)
    "user@-example.com"                        -> False  (domain label starts with '-')
    "user@example-.com"                        -> False  (domain label ends with '-')
    "user@examplecom"                          -> False  (domain has no dot: only one label)
    "user@example.c"                           -> False  (TLD too short, 1 char)
    "user@example.c0m"                         -> False  (TLD contains a digit)
    "user@example.co-m"                        -> False  (TLD contains a hyphen)
    "us er@example.com"                        -> False  (space -- not an allowed local char)
    "user@exa mple.com"                        -> False  (space -- not an allowed domain char)
    "user#name@example.com"                    -> False  ('#' not in the allowed local charset)
"""
from __future__ import annotations

CANONICAL_TEST_PATH = "tests/test_email_validator.py"

_TEST_FILE_CONTENT = '''\
from email_validator import is_valid_email


def test_minimal_valid_address():
    assert is_valid_email("a@b.co") is True


def test_typical_valid_address():
    assert is_valid_email("john.doe@example.com") is True


def test_valid_with_special_local_chars_and_three_domain_labels():
    assert is_valid_email("user_name+tag@sub-domain.example.org") is True


def test_valid_with_multiple_non_consecutive_local_dots():
    assert is_valid_email("a.b.c@example.co") is True


def test_valid_with_digits_in_local_and_non_tld_label():
    assert is_valid_email("x99@ab12.io") is True


def test_empty_string():
    assert is_valid_email("") is False


def test_missing_at_sign():
    assert is_valid_email("noatsign.example.com") is False


def test_two_at_signs():
    assert is_valid_email("two@at@signs.com") is False


def test_empty_local_part():
    assert is_valid_email("@example.com") is False


def test_empty_domain():
    assert is_valid_email("user@") is False


def test_local_starts_with_dot():
    assert is_valid_email(".user@example.com") is False


def test_local_ends_with_dot():
    assert is_valid_email("user.@example.com") is False


def test_local_has_consecutive_dots():
    assert is_valid_email("us..er@example.com") is False


def test_domain_leading_dot():
    assert is_valid_email("user@.example.com") is False


def test_domain_trailing_dot():
    assert is_valid_email("user@example.com.") is False


def test_domain_consecutive_dots():
    assert is_valid_email("user@example..com") is False


def test_domain_label_starts_with_hyphen():
    assert is_valid_email("user@-example.com") is False


def test_domain_label_ends_with_hyphen():
    assert is_valid_email("user@example-.com") is False


def test_domain_has_no_dot():
    assert is_valid_email("user@examplecom") is False


def test_tld_too_short():
    assert is_valid_email("user@example.c") is False


def test_tld_contains_digit():
    assert is_valid_email("user@example.c0m") is False


def test_tld_contains_hyphen():
    assert is_valid_email("user@example.co-m") is False


def test_space_in_local_part():
    assert is_valid_email("us er@example.com") is False


def test_space_in_domain():
    assert is_valid_email("user@exa mple.com") is False


def test_disallowed_local_character():
    assert is_valid_email("user#name@example.com") is False
'''

# Each keystone case probes a genuinely distinct rule/state-transition: local dot placement
# (start/end/consecutive), domain label emptiness from dot placement (leading/trailing/
# consecutive), domain label hyphen placement (start/end), the "at least two labels" rule,
# the three TLD-specific rules (length, digits, hyphens), and two valid-but-easy-to-overreject
# cases (special local characters spread across three domain labels; digits allowed in a
# non-TLD label but NOT the TLD -- catches an implementation that wrongly applies the
# letters-only TLD rule to every label) -- these are exactly the rules a single naive "looks
# vaguely like an email" check (e.g. just "contains one @ and a dot after it") would get
# wrong. The two simplest valid examples, the completely-missing-@ and multiple-@ structural
# cases, and the plain-disallowed-character cases are bonus credit only, not keystone, since a
# correct or even a fairly naive implementation is unlikely to get those specific ones wrong.
KEYSTONE_TEST_IDS = [
    f"{CANONICAL_TEST_PATH}::test_valid_with_special_local_chars_and_three_domain_labels",
    f"{CANONICAL_TEST_PATH}::test_valid_with_multiple_non_consecutive_local_dots",
    f"{CANONICAL_TEST_PATH}::test_valid_with_digits_in_local_and_non_tld_label",
    f"{CANONICAL_TEST_PATH}::test_local_starts_with_dot",
    f"{CANONICAL_TEST_PATH}::test_local_ends_with_dot",
    f"{CANONICAL_TEST_PATH}::test_local_has_consecutive_dots",
    f"{CANONICAL_TEST_PATH}::test_domain_leading_dot",
    f"{CANONICAL_TEST_PATH}::test_domain_trailing_dot",
    f"{CANONICAL_TEST_PATH}::test_domain_consecutive_dots",
    f"{CANONICAL_TEST_PATH}::test_domain_label_starts_with_hyphen",
    f"{CANONICAL_TEST_PATH}::test_domain_label_ends_with_hyphen",
    f"{CANONICAL_TEST_PATH}::test_domain_has_no_dot",
    f"{CANONICAL_TEST_PATH}::test_tld_too_short",
    f"{CANONICAL_TEST_PATH}::test_tld_contains_digit",
    f"{CANONICAL_TEST_PATH}::test_tld_contains_hyphen",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c32",
        "title": "email-fsm-validator",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `email_validator.py` that defines a function "
        "`is_valid_email(s: str) -> bool`.\n\n"
        "This validates a SIMPLIFIED email-address format (not full RFC 5322) against the "
        "grammar below. Implement it as an EXPLICIT FINITE STATE MACHINE — walk the string "
        "(or its LOCAL/DOMAIN halves) one character at a time, maintaining an explicit state "
        "variable that tracks what you've seen so far and what characters are legal next, "
        "rather than building one large regular expression that tries to match the whole "
        "grammar in a single pattern. (You may still use a regex for a trivial, single-rule "
        "helper — e.g. checking whether one character is a letter — but the overall "
        "start-to-end validation must be driven by your own explicit state-machine logic, "
        "not by handing the whole string to one big compiled pattern.)\n\n"
        "## Grammar\n\n"
        "`LOCAL \"@\" DOMAIN`\n\n"
        "- `LOCAL`: one or more characters, each either a letter (`a`-`z`, `A`-`Z`), a digit "
        "(`0`-`9`), or one of `.` `_` `%` `+` `-`. `LOCAL` must not START with `.`, must not "
        "END with `.`, and must not contain two consecutive dots (`..`) anywhere.\n"
        "- `DOMAIN`: two or more \"labels\" separated by single dots (`.`). Each label is one "
        "or more characters, each either a letter, a digit, or `-`; a label must not START "
        "or END with `-`, and must be non-empty (so a `DOMAIN` that starts with `.`, ends "
        "with `.`, or contains `..` is invalid — any of those produces an empty label). The "
        "LAST label (the TLD, e.g. `com` in `example.com`) has two EXTRA requirements beyond "
        "the ordinary label rules: it must consist ONLY of letters (no digits, no `-` at "
        "all, anywhere in it), and it must be at least 2 characters long.\n"
        "- The overall string `s` must contain EXACTLY one `@` character. Any character in "
        "`s` that isn't a letter, digit, `.`, `_`, `%`, `+`, `-`, or the single `@` makes the "
        "string invalid (e.g. a space anywhere makes it invalid).\n"
        "- \"Letter\" means any of the 26 ASCII letters, uppercase or lowercase.\n\n"
        "Worked examples: `is_valid_email(\"a@b.co\")` -> `True` (shortest valid shape: one "
        "local char, one non-TLD label char, a 2-letter TLD). "
        "`is_valid_email(\"user@example..com\")` -> `False` (the doubled dot in the domain "
        "produces an empty label between them). `is_valid_email(\"user@example.c0m\")` -> "
        "`False` (the TLD `c0m` contains a digit, which the ordinary label rules would "
        "otherwise allow — this is the TLD-specific extra restriction).\n\n"
        "There is no visible test file for this task. Write email_validator.py implementing "
        "is_valid_email per the grammar above, structured as an explicit state machine "
        "(e.g. separate state-tracking passes over the local part and each domain label, or "
        "one combined character walk with an explicit state variable — your choice, as long "
        "as it's explicit control flow rather than one big regex). Use run_python to "
        "sanity-check it yourself against a handful of valid and invalid addresses (including "
        "at least one violation of each rule: local leading/trailing/consecutive dots, domain "
        "leading/trailing/consecutive dots, domain label leading/trailing hyphen, a domain "
        "with only one label, and a TLD that's too short / has a digit / has a hyphen) before "
        "finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {CANONICAL_TEST_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "email_validator", "functions": ["is_valid_email"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write email_validator.py implementing is_valid_email(s: str) -> bool "
                    "for this simplified grammar: LOCAL '@' DOMAIN, where s must contain "
                    "EXACTLY one '@'.\n\n"
                    "LOCAL (the part before '@'): one or more chars from letters/digits/"
                    "'.'/'_'/'%'/'+'/'-' ; must not start with '.', must not end with '.', "
                    "must not contain '..'  (walk it char by char with an explicit state "
                    "tracking whether the previous char was a dot, to catch '..' and "
                    "leading/trailing dots).\n\n"
                    "DOMAIN (the part after '@'): split on '.' into labels; there must be "
                    "at least 2 labels (a domain with no dot at all is invalid). Each label: "
                    "one or more chars from letters/digits/'-' ; must not start or end with "
                    "'-' ; must be non-empty (an empty label happens when DOMAIN starts with "
                    "'.', ends with '.', or contains '..' -- reject all of those). "
                    "ADDITIONALLY, the LAST label (the TLD) must be ALL LETTERS (no digits, "
                    "no hyphens at all) and at least 2 characters long -- this is stricter "
                    "than the ordinary label rule, check it separately.\n\n"
                    "Implement this as an explicit finite-state-machine style walk (an "
                    "explicit state variable per character, e.g. tracking 'was the previous "
                    "character a dot') rather than one single big regex covering the whole "
                    "grammar in one pattern -- small regexes for single-character-class "
                    "checks are fine.\n\n"
                    "Use write_file to create it. There is no pre-existing test file -- use "
                    "run_python to check several valid addresses (e.g. 'a@b.co', "
                    "'john.doe@example.com') return True, and several invalid ones each "
                    "violating exactly one rule (e.g. '.user@example.com', "
                    "'user@example..com', 'user@-example.com', 'user@examplecom', "
                    "'user@example.c', 'user@example.c0m', 'user@example.co-m') return "
                    "False. Fix issues with patch_file/run_python until confident, then "
                    "finish."
                ),
                "expect": "email_validator.py written implementing is_valid_email as an "
                          "explicit state machine per the grammar",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm email_validator.py exists and defines is_valid_email.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["email_validator.py"]},
    }
