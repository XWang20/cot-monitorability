# Adapted from semsorock/inspect-evals-monitor-bench at 9452b3ddcc1be81c4322e141dceb6f6c69b49434 (MIT); see NOTICE.
"""Reference generation defaults for the MonitorBench monitor role.

Inspect does not propagate evaluated-model sampling settings to role models.
This helper fills only monitor settings that the caller did not configure,
while deliberately leaving ``seed`` unset: upstream produces four monitor
rollouts in one seeded vLLM request, whereas this port makes four separate
requests and reusing one seed can collapse their diversity.
"""

from inspect_ai.model import GenerateConfig, Model, get_model
REFERENCE_TEMPERATURE = 0.6
REFERENCE_TOP_P = 0.9
MONITOR_MAX_TOKENS = 16768


def get_monitor_model() -> Model:
    """Use the configured monitor, otherwise explicitly reuse the evaluated model."""
    # Called lazily by scorers, never during task construction. The location
    # rule does not follow calls into shared helpers.
    return get_model(role="monitor", default="openai/deepseek-v4.1-flash-1017")


def monitor_role_config(model: Model) -> GenerateConfig:
    """Fill absent monitor-role settings with the upstream sampling defaults."""
    configured = model.config
    return GenerateConfig(
        temperature=(REFERENCE_TEMPERATURE if configured.temperature is None else None),
        top_p=REFERENCE_TOP_P if configured.top_p is None else None,
        max_tokens=MONITOR_MAX_TOKENS if configured.max_tokens is None else None,
    )
