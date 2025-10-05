import pytest
from _pytest import mark

from app.prompt_builder import PromptBuilder

mandate = "Find job listings"
short_term = "1. Checked previous job boards"
notes = "Remember to check TechCorp"
retrieved_long_term = "Previous job listings: Google, Amazon"
observations = "<html>Job page content</html>"

pb = PromptBuilder(
    mandate=mandate,
    short_term_summary=short_term,
    notes=notes,
    retrieved_long_term=retrieved_long_term,
    observations=observations
)

def test_prompt_builder_basic():
    assert pb.mandate == mandate
    assert pb.short_term_summary == short_term
    assert pb.notes == notes
    assert pb.retrieved_long_term == retrieved_long_term
    assert pb.observations == observations

@pytest.mark.parametrize("snippet", [mandate, short_term, notes, retrieved_long_term, observations])
def test_prompt_builder_snippets(snippet):
    assert snippet in pb._build_user_message()