import asyncio
import uuid
import pytest

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.message_contract import KeyNames


@pytest.mark.asyncio
async def test_agent_container_roundtrip_via_rabbitmq():
    """
    Publish a task to the real queues and observe status stream from the live agent.
    :return None: Nothing is returned
    """
    cfg = ConnectorConfig()
    rmq = ConnectorRabbitMQ(cfg)
    await rmq.connect()

    task_id = str(uuid.uuid4())
    codeword = "roundtrip"
    mandate = f"Respond once using the word '{codeword}' and then exit."

    seen: list[dict] = []

    async def collect_status(message: dict):
        if message.get(KeyNames.CORRELATION_ID) not in {None, task_id} and message.get(KeyNames.TASK_ID) not in {None, task_id}:
            return
        seen.append(message)

    consumer = asyncio.create_task(rmq.consume_status_updates(collect_status))

    payload = {
        KeyNames.MANDATE: mandate,
        KeyNames.MAX_TICKS: 3,
        KeyNames.CORRELATION_ID: task_id,
        KeyNames.TASK_ID: task_id,
    }
    await rmq.publish_task(task_id=task_id, payload=payload)

    final = None
    for _ in range(600):
        if any(msg.get(KeyNames.TYPE) in {"completed", "error"} for msg in seen):
            final = next(msg for msg in seen[::-1] if msg.get(KeyNames.TYPE) in {"completed", "error"})
            break
        await asyncio.sleep(0.2)

    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    await rmq.disconnect()

    assert seen, "No status messages received from agent"
    assert any(msg.get(KeyNames.TYPE) not in {"completed", "error"} for msg in seen), "No intermediate status messages"
    assert final is not None and final.get(KeyNames.TYPE) == "completed", f"Final status not completed: {final}"
    assert codeword in str(final.get(KeyNames.MANDATE, "")), "Mandate not echoed in completed message"

