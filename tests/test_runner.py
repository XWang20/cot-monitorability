import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner_env(tmp_path):
    # Resolve task signatures normally, but capture the CLI instead of calling APIs.
    executable = tmp_path / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[3] == 'python':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[4:]])\n"
        "print(json.dumps(sys.argv[3:]))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    env = os.environ.copy()
    env.pop("REASONING_EFFORT", None)
    env.update(
        PATH=str(tmp_path) + os.pathsep + env["PATH"],
        OPENAI_BASE_URL="http://127.0.0.1:8000/v1",
        OPENAI_API_KEY="a",
        JUDGE_MODEL="mockllm/judge",
        COT_USE_LITELLM_COMPAT="auto",
    )
    return env


@pytest.mark.parametrize(
    "task,effort",
    [
        ("cot_controllability_qa", "high"),
        ("openai_monitorability_process", "medium"),
        ("openai_monitorability_intervention", "medium"),
        ("cot_immediacy_2510", "high"),
        ("ncri_bench", None),
        ("monitorbench", "high"),
    ],
)
def test_runner_preserves_task_settings_without_model_calls(runner_env, task, effort):
    output = subprocess.check_output(
        [str(ROOT / "scripts/run_eval.sh"), task, "mockllm/actor", "--limit", "1"],
        env=runner_env,
        cwd="/",
        text=True,
    )
    args = json.loads(output)
    assert args[:3] == ["inspect", "eval", f"cot_monitorability/{task}"]
    assert args[-2:] == ["--limit", "1"]
    body = json.loads(args[args.index("--extra-body") + 1])
    if effort:
        assert body["reasoning"]["effort"] == effort
        assert f"target_reasoning_effort={effort}" in args
    else:
        assert "--reasoning-effort" not in args
        assert "reasoning" not in body
    roles = [
        json.loads(args[i + 1].split("=", 1)[1])
        for i, arg in enumerate(args)
        if arg == "--model-role"
    ]
    assert len(roles) == 2
    assert all(role["model"] == "mockllm/judge" for role in roles)
    assert all(role["extra_body"]["reasoning"]["effort"] == "high" for role in roles)


def test_runner_disables_proxy_fields_for_external_endpoint(runner_env):
    runner_env["OPENAI_BASE_URL"] = "https://example.test/v1"
    output = subprocess.check_output(
        [str(ROOT / "scripts/run_eval.sh"), "cot_controllability_qa", "mockllm/actor"],
        env=runner_env,
        text=True,
    )
    assert "--extra-body" not in json.loads(output)


def test_runner_rejects_cot_effort_override_for_ncri(runner_env):
    runner_env["REASONING_EFFORT"] = "high"
    result = subprocess.run(
        [str(ROOT / "scripts/run_eval.sh"), "ncri_bench", "mockllm/actor"],
        env=runner_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no-CoT" in result.stderr
