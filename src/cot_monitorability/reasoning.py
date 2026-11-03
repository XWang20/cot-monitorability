"""Require readable full reasoning when scoring CoT-dependent tasks."""

from inspect_ai.model import ChatMessageAssistant, ContentReasoning


def full_reasoning(message: ChatMessageAssistant) -> str:
    blocks = (
        [block for block in message.content if isinstance(block, ContentReasoning)]
        if isinstance(message.content, list)
        else []
    )
    parts = [
        block.reasoning for block in blocks if block.reasoning and not block.redacted
    ]
    if any(
        block.redacted or (block.summary and not block.reasoning) for block in blocks
    ):
        raise ValueError(
            "Full reasoning is unavailable: the provider returned only a summary "
            "or redacted reasoning. Use a provider that exposes the complete trace."
        )
    return "\n".join(parts).strip()
