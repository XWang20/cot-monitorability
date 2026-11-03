"""Request-time RMB accounting and conservative reservations for experiment runners."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from filelock import FileLock


class BudgetExceeded(RuntimeError):
    pass


def estimate_cost(usage, started_at, pricing):
    local = datetime.fromisoformat(started_at).astimezone(ZoneInfo(pricing["timezone"]))
    minute = local.strftime("%H:%M")
    peak = local.weekday() < 5 and any(
        start <= minute < end for start, end in pricing["weekday_peak_windows"]
    )
    rates = pricing["peak_per_million"] if peak else pricing["offpeak_per_million"]
    result = {"cost": None, "rates": rates, "period": "peak" if peak else "offpeak"}
    if not usage or any(
        usage.get(k) is None for k in ("prompt_tokens", "completion_tokens")
    ):
        return result
    prompt = int(usage["prompt_tokens"])
    cached = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    output = int(usage["completion_tokens"])
    if prompt == 0 and output == 0:
        return result
    uncached = max(0, prompt - cached)
    return {
        **result,
        "cost": (
            uncached * rates["input"]
            + cached * rates["cached_input"]
            + output * rates["output"]
        )
        / 1_000_000,
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
    }


def read_jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


class RequestBudget:
    """Reserve uncached-input plus maximum-output cost before dispatching an RPC.

    Missing responses/usage retain their reservation. Cache hits make no RPC and
    therefore neither reserve nor charge. All worker processes share this ledger.
    """

    def __init__(
        self, directory: Path, limit: float, baseline: float = 0, uncertainty: float = 0
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit, self.baseline, self.uncertainty = limit, baseline, uncertainty
        self.lock = FileLock(str(self.directory / "budget.lock"))
        self.starts = self.directory / "requests_started.jsonl"
        self.responses = self.directory / "requests.jsonl"
        settings = {"limit": limit, "baseline": baseline, "uncertainty": uncertainty}
        with self.lock:
            path = self.directory / "budget.json"
            if path.exists() and json.loads(path.read_text()) != settings:
                raise ValueError("Budget settings differ from the existing experiment")
            if not path.exists():
                path.write_text(json.dumps(settings, indent=2) + "\n")

    @staticmethod
    def _append(path, value):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()

    def _snapshot(self):
        starts, responses = read_jsonl(self.starts), read_jsonl(self.responses)
        settled = {
            row["request_id"] for row in responses if row.get("cost") is not None
        }
        unresolved = [row for row in starts if row["request_id"] not in settled]
        known = sum(row["cost"] for row in responses if row.get("cost") is not None)
        reserved = sum(row["reserved_cost"] for row in unresolved)
        return {
            "currency": "CNY",
            "limit": self.limit,
            "baseline_known_cost": self.baseline,
            "baseline_uncertainty_reserve": self.uncertainty,
            "new_known_cost": known,
            "known_total": self.baseline + known,
            "outstanding_reservations": reserved,
            "available": self.limit
            - self.baseline
            - self.uncertainty
            - known
            - reserved,
            "unresolved": unresolved,
            "responses": responses,
        }

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    async def reserve(self, body, pricing, context):
        maximum = body.get("max_completion_tokens", body.get("max_tokens"))
        if not isinstance(maximum, int) or maximum <= 0:
            raise ValueError(
                "Budgeted requests require an explicit positive output limit"
            )
        # UTF-8 bytes bound normal text token counts conservatively; include framing.
        prompt_bound = (
            len(json.dumps(body.get("messages", []), ensure_ascii=False).encode())
            + 2048
        )
        while True:
            started = datetime.now(timezone.utc).isoformat()
            rate = estimate_cost(None, started, pricing)["rates"]
            upper = (
                math.ceil(prompt_bound * rate["input"] + maximum * rate["output"])
                / 1_000_000
            )
            record = {
                **context,
                "request_id": str(uuid.uuid4()),
                "started_at": started,
                "reserved_cost": upper,
                "max_output_tokens": maximum,
            }
            with self.lock:
                current = self._snapshot()
                if upper <= current["available"]:
                    self._append(self.starts, record)
                    return record
                now = datetime.now(timezone.utc)
                active = any(
                    (now - datetime.fromisoformat(row["started_at"])).total_seconds()
                    < 660
                    for row in current["unresolved"]
                )
                if not active:
                    raise BudgetExceeded(
                        f"RMB budget cannot reserve {upper:.4f}; available {current['available']:.4f}"
                    )
            await asyncio.sleep(1)

    def settle(self, reservation, usage, pricing, status):
        record = {
            **reservation,
            **estimate_cost(usage, reservation["started_at"], pricing),
            "status": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self._append(self.responses, record)
        return record

    def hooks(self, pricing, context, expected_effort):
        async def before(request):
            if not request.url.path.endswith("/chat/completions"):
                raise ValueError(
                    "This runner budgets non-streaming Chat Completions requests only"
                )
            body = json.loads(request.content)
            if body.get(
                "reasoning_effort"
            ) != expected_effort or "reasoning_effort" not in body.get(
                "allowed_openai_params", []
            ):
                raise ValueError(
                    "Request reasoning effort or proxy passthrough differs from the protocol"
                )
            if expected_effort == "none" and (body.get("reasoning") or {}).get(
                "effort"
            ) not in (None, "none"):
                raise ValueError("NCRI request contains conflicting reasoning settings")
            request.extensions["experiment_reservation"] = await self.reserve(
                body, pricing, context()
            )

        async def after(response):
            await response.aread()
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            self.settle(
                response.request.extensions["experiment_reservation"],
                payload.get("usage"),
                pricing,
                response.status_code,
            )

        return {"request": [before], "response": [after]}
