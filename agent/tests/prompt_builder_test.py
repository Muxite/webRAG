import pytest
from _pytest import mark

from app.prompt_builder import PromptBuilder

mandate = "Find job listings"
short_term = ["A", "B", "C"]
notes = "Remember to check TechCorp"
retrieved_long_term = ["A"]
observations = "<html>Job page content</html>"

added_history = "D"
replacement_note = "Checked the site"

pb = PromptBuilder(
    mandate=mandate,
    short_term_summary=short_term,
    notes=notes,
    retrieved_long_term=retrieved_long_term,
    observations=observations
)

@pytest.mark.parametrize("snippet", [mandate, short_term[0], replacement_note, retrieved_long_term[0], observations,
                                     added_history])
def test_prompt_builder_snippets(snippet):
    pb.update_notes(replacement_note)
    pb.add_history_entry(added_history)
    assert snippet in pb._build_user_message()
