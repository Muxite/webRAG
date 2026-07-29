"""P0b tool + verifier unit tests (shell / web / vector-memory / verify). No LLM."""
from localagent.tools.base import ToolContext
from localagent.tools.memory import InMemoryBackend, VectorMemoryStore
from localagent.tools.shell import ShellTool
from localagent.tools.web import WebReadTool, _regex_extract
from localagent.verify import (contains_number, evidence_first, file_contains,
                               schema_valid)


# --- shell (read-only allow-listed DSL) ---------------------------------------
def test_shell_count_lines(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\nz\n")
    r = ShellTool().execute({"op": "count_lines", "file": str(tmp_path / "a.txt")},
                            ToolContext(workdir=tmp_path))
    assert r.ok and "3" in r.data["stdout"]


def test_shell_grep_pattern(tmp_path):
    (tmp_path / "a.txt").write_text("foo\nbar\nfoo\n")
    r = ShellTool().execute({"op": "count_lines", "file": str(tmp_path / "a.txt"), "pattern": "foo"},
                            ToolContext(workdir=tmp_path))
    assert r.ok and "2" in r.data["stdout"]


def test_shell_find(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "needle.txt").write_text("x")
    r = ShellTool().execute({"op": "find", "name": "needle.txt"}, ToolContext(workdir=tmp_path))
    assert r.ok and "needle.txt" in r.data["stdout"]


def test_shell_path_escape_refused(tmp_path):
    r = ShellTool().execute({"op": "count_lines", "file": "/etc/passwd"},
                            ToolContext(workdir=tmp_path))
    assert not r.ok and r.error == "path"


# --- web (decoupled via fakes) ------------------------------------------------
def _fake_search(q, k):
    return [{"title": "Quesnel Lake", "url": "http://x/q", "snippet": "depth 511"}]


def _fake_fetch(url):
    return "<html><body>Max depth <b>511</b> m</body></html>"


def test_web_search_registers_url_entities(tmp_path):
    tool = WebReadTool(_fake_search, _fake_fetch)
    r = tool.execute({"op": "search", "query": "quesnel"}, ToolContext(workdir=tmp_path))
    assert r.ok and r.new_entities and r.new_entities[0][0] == "url"


def test_web_read_extracts_text(tmp_path):
    tool = WebReadTool(_fake_search, _fake_fetch)
    r = tool.execute({"op": "read", "url": "http://x/q"}, ToolContext(workdir=tmp_path))
    assert r.ok and "511" in r.data["text"]


def test_regex_extract_strips_tags():
    assert _regex_extract("<p>hello <b>world</b></p>") == "hello world"


# --- vector memory (InMemoryBackend) ------------------------------------------
def test_vector_memory_roundtrip():
    s = VectorMemoryStore("user1", InMemoryBackend())
    s.remember("the deadline is October 4")
    assert any("October" in h for h in s.recall("deadline"))


def test_vector_memory_identity_isolation():
    backend = InMemoryBackend()
    VectorMemoryStore("u1", backend).remember("secret alpha token")
    assert VectorMemoryStore("u2", backend).recall("secret") == []


# --- verifiers ----------------------------------------------------------------
def test_evidence_first_and_number():
    assert evidence_first("511", "max depth is 511 m")
    assert not evidence_first("999", "max depth is 511 m")
    assert contains_number("value 511 m", "511")
    assert not contains_number("5110", "511")


def test_schema_valid():
    assert schema_valid('{"a": 1}', {"a": "number"}).passed
    assert not schema_valid('{"a": "1"}', {"a": "number"}).passed        # stringified number
    assert not schema_valid("not json", {"a": "number"}).passed


def test_file_contains(tmp_path):
    (tmp_path / "f.txt").write_text("hello")
    assert file_contains(tmp_path, "f.txt", "hello").passed
    assert not file_contains(tmp_path, "f.txt", "bye").passed
