"""Daily token / cost budget tracker.

Tracks per-day spending against a configurable cap. The cap is in USD by
default but you can also set a soft token cap. State persists to disk so
restarts don't lose accounting; date rollover (UTC) auto-resets counters.

Usage in the daemon:

    budget = DailyBudget("state/daily_budget.json", cap_usd=5.0)
    if budget.would_exceed(estimated_cost=0.005):
        seconds = budget.seconds_until_reset()
        log.info(f"Daily cap hit; sleeping {seconds:.0f}s until next UTC day")
        time.sleep(seconds)
    # ... make the API call ...
    budget.record(cost_usd=0.0046, tokens_in=12, tokens_out=200)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass
class DayLedger:
    date: str          # ISO YYYY-MM-DD (UTC)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_in_cached: int = 0
    tokens_out: int = 0
    n_calls: int = 0
    n_calls_blocked: int = 0


class DailyBudget:
    """Per-day USD/token tracker with disk persistence and UTC rollover."""

    def __init__(
        self,
        path: str | Path,
        cap_usd: float = 5.0,
        cap_tokens: Optional[int] = None,
    ):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.cap_tokens = cap_tokens
        self._ledger = self._load()

    def _today_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load(self) -> DayLedger:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            ledger = DayLedger(**data)
            # Rollover if the persisted date isn't today (UTC).
            if ledger.date != self._today_utc():
                ledger = DayLedger(date=self._today_utc())
                self._flush(ledger)
            return ledger
        ledger = DayLedger(date=self._today_utc())
        self._flush(ledger)
        return ledger

    def _flush(self, ledger: DayLedger) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(ledger), indent=2))
        tmp.replace(self.path)

    def _refresh_for_today(self) -> None:
        if self._ledger.date != self._today_utc():
            self._ledger = DayLedger(date=self._today_utc())
            self._flush(self._ledger)

    def remaining_usd(self) -> float:
        self._refresh_for_today()
        return max(0.0, self.cap_usd - self._ledger.cost_usd)

    def would_exceed(self, estimated_cost_usd: float) -> bool:
        return self.remaining_usd() < estimated_cost_usd

    def record(
        self,
        cost_usd: float,
        tokens_in: int = 0,
        tokens_in_cached: int = 0,
        tokens_out: int = 0,
        blocked: bool = False,
    ) -> None:
        self._refresh_for_today()
        self._ledger.cost_usd += cost_usd
        self._ledger.tokens_in += tokens_in
        self._ledger.tokens_in_cached += tokens_in_cached
        self._ledger.tokens_out += tokens_out
        if blocked:
            self._ledger.n_calls_blocked += 1
        else:
            self._ledger.n_calls += 1
        self._flush(self._ledger)

    def seconds_until_reset(self) -> float:
        """Seconds remaining until next UTC midnight."""
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Add one day
        from datetime import timedelta
        tomorrow = tomorrow + timedelta(days=1)
        return max(60.0, (tomorrow - now).total_seconds())

    def snapshot(self) -> dict:
        self._refresh_for_today()
        return {
            "date": self._ledger.date,
            "cap_usd": self.cap_usd,
            "spent_usd": round(self._ledger.cost_usd, 4),
            "remaining_usd": round(self.remaining_usd(), 4),
            "calls": self._ledger.n_calls,
            "calls_blocked": self._ledger.n_calls_blocked,
            "tokens_in": self._ledger.tokens_in,
            "tokens_in_cached": self._ledger.tokens_in_cached,
            "tokens_out": self._ledger.tokens_out,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(EXPERIMENT_DIR / "state" / "daily_budget.json"))
    ap.add_argument("--cap-usd", type=float, default=5.0)
    args = ap.parse_args()
    b = DailyBudget(args.state, cap_usd=args.cap_usd)
    print(json.dumps(b.snapshot(), indent=2))
