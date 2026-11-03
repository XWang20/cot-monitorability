import json
from types import SimpleNamespace

import httpx
import pytest
from inspect_ai.model import ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.scorer import Target

from cot_monitorability.immediacy import components as immediacy


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", ["deepseek-v4.1-flash-1017", "kimi-k3.1-pro"])
async def test_judge_sampling_parameters_reach_compatible_http_request(
    monkeypatch, checkpoint
):
    requests = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": checkpoint,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"legibility_score":3,"coverage_score":4}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        models = []

        def judge_model(*, role, default, config):
            assert role == "judge"
            model = get_model(
                default,
                config=config,
                base_url="https://example.test/v1",
                api_key="test-key",
                http_client=client,
                memoize=False,
            )
            models.append(model)
            return model

        monkeypatch.setattr(immediacy, "get_model", judge_model)
        sample = SimpleNamespace(
            input_text="Choose A.",
            output=ModelOutput.from_content(
                "mockllm/actor",
                [
                    ContentReasoning(reasoning="A readable reasoning step."),
                    ContentText(text="ANSWER: A"),
                ],
            ),
        )
        kwargs = (
            {}
            if checkpoint == "deepseek-v4.1-flash-1017"
            else {"judge_model": f"openai-api/openai/{checkpoint}"}
        )
        result = await immediacy.cot_immediacy_judge_scorer(**kwargs)(
            sample, Target("A")
        )
        for model in models:
            await model.__aexit__(None, None, None)

    assert result.value["legibility_pct"] == 75
    assert len(requests) == 1
    body = requests[0]
    assert body["model"] == checkpoint
    assert body["temperature"] == 0
    assert body["max_tokens"] == 8192
    assert body["reasoning_effort"] == "high"
    assert "max_completion_tokens" not in body
