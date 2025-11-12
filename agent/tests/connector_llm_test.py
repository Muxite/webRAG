import pytest_asyncio
import pytest
from app.connector_llm import ConnectorLLM
from app.connector_config import ConnectorConfig
from app.prompt_builder import PromptBuilder, build_payload
import os
import time

@pytest_asyncio.fixture
def connector_config():
    return ConnectorConfig()

@pytest_asyncio.fixture()
def connector(connector_config):
    return ConnectorLLM(connector_config)

@pytest.mark.asyncio
async def test_llm_query(connector: ConnectorLLM):
    """Test a basic LLM query returns a valid, non-empty response."""
    prompt = "Say 'pong' if you can read this."
    builder = PromptBuilder(observations=prompt)

    llm_response = await connector.query_llm(build_payload(builder.build_messages(), json_mode=False))

    assert llm_response is not None, "LLM query returned None"
    assert isinstance(llm_response, str)
    assert "pong" in llm_response.lower(), "LLM response did not contain 'pong'"

@pytest.mark.asyncio
async def test_llm_latency_writing_curve(connector):
    word_counts = [25, 50, 100, 200, 200, 400]
    results = []

    for word_count in word_counts:
        prompt = (
            f"Write approximately {word_count} words about the history of food in society. "
            f"Do not include explanations about being an AI; just reply with text ~{word_count} words."
        )
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": int(word_count * 2)
        }

        t0 = time.perf_counter()
        llm_response = await connector.query_llm(payload)
        t1 = time.perf_counter()

        assert llm_response and isinstance(llm_response, str) and llm_response.strip()

        elapsed = t1 - t0
        realized_words = len(llm_response.strip().split())
        words_per_sec = realized_words / elapsed if elapsed > 0 else float("inf")

        if word_count > 0:
            assert (abs(
                realized_words - word_count) / word_count) < 0.25, "Response too far away from requested word count"

        results.append({
            "requested": word_count,
            "realized": realized_words,
            "elapsed": elapsed,
            "wps": words_per_sec
        })
        print(
            f"requested words: {word_count}, written words: {realized_words}, "
            f"elapsed seconds: {elapsed:.3f}, words per second: {words_per_sec:.2f}"
        )


@pytest.mark.asyncio
async def test_llm_latency_reading_curve(connector):
    script_dir = os.path.dirname(__file__)
    text_path = os.path.join(script_dir, "long_text.txt")
    with open(text_path, "r", encoding="utf-8") as f:
        LONG_TEXT = f.read()

    word_counts = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
    results = []
    long_word_list = LONG_TEXT.split()

    for word_count in word_counts:
        input_text = " ".join(long_word_list[:word_count])
        prompt = (
            f"{input_text}\n\nSummarize the text as best you can."
        )

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        }

        t0 = time.perf_counter()
        llm_response = await connector.query_llm(payload)
        t1 = time.perf_counter()

        assert llm_response and isinstance(llm_response, str) and llm_response.strip()

        elapsed = t1 - t0
        realized_words_in_prompt = len(input_text.strip().split())

        results.append({
            "prompt_words": realized_words_in_prompt,
            "elapsed": elapsed,
        })
        print(
            f"prompt words: {realized_words_in_prompt}, "
            f"elapsed seconds: {elapsed:.3f}, response: {llm_response.strip()[:100]}..."
        )