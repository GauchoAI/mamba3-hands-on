"""Unified run facade for Lab live telemetry and archive persistence.

`LabRun` is intentionally a compatibility layer over the existing v1
Firebase/Hugging Face implementation. It does not change Firebase paths,
stream metadata, manifest files, or bucket layout. New trainers can use one
object while the populated dashboard data and current readers keep working.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .experiment_pusher import ExperimentPusher, DEFAULT_FIREBASE_URL

try:
    from .cloud_archive import CloudArchive
    _HAS_ARCHIVE = True
except ImportError:  # pragma: no cover - exercised only in lean installs
    CloudArchive = None  # type: ignore[assignment]
    _HAS_ARCHIVE = False


def _config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return {
            k: v for k, v in vars(config).items()
            if not k.startswith("_")
        }
    return {"value": str(config)}


class LabRun:
    """One user-facing handle for a training run.

    The facade delegates to:
    - `ExperimentPusher` for Firebase live state and local stream JSONL.
    - `CloudArchive` for Hugging Face bucket sync.

    It keeps both optional so tests, local dry-runs, and machines without
    credentials can still execute the exact same trainer code.
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        run_id: str,
        kind: str,
        config: Any,
        out_dir: str | Path,
        firebase_url: str = DEFAULT_FIREBASE_URL,
        live_enabled: bool = True,
        archive_enabled: bool = True,
        archive_sync_every_s: int | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.kind = kind
        self.config = _config_dict(config)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.pusher: ExperimentPusher | None = None
        if live_enabled:
            self.pusher = ExperimentPusher(
                experiment_id=experiment_id,
                run_id=run_id,
                kind=kind,
                config=self.config,
                outbox_dir=self.out_dir,
                firebase_url=firebase_url,
            )

        self.archive = None
        if archive_enabled and _HAS_ARCHIVE and CloudArchive is not None:
            kwargs: dict[str, Any] = {
                "experiment_kind": kind,
                "run_name": run_id,
                "local_dir": self.out_dir,
            }
            if archive_sync_every_s is not None:
                kwargs["sync_every_s"] = archive_sync_every_s
            self.archive = CloudArchive(**kwargs)

    @property
    def live(self) -> bool:
        return self.pusher is not None

    @property
    def archived(self) -> bool:
        return self.archive is not None

    def start(
        self,
        *,
        name: str,
        purpose: str = "",
        hypothesis: str = "",
        gpu: int | None = None,
        experiment_extra: dict[str, Any] | None = None,
        run_extra: dict[str, Any] | None = None,
    ) -> None:
        """Declare experiment and run metadata using the existing v1 schema."""
        if self.pusher is None:
            return
        self.pusher.declare_experiment(
            name=name,
            hypothesis=hypothesis,
            extra=experiment_extra,
        )
        self.pusher.declare_run(
            purpose=purpose,
            gpu=gpu,
            extra=run_extra,
        )

    def declare_stream(self, name: str, **kwargs: Any) -> None:
        if self.pusher is not None:
            self.pusher.declare_stream(name, **kwargs)

    def stream(self, name: str, record: dict[str, Any]) -> None:
        if self.pusher is not None:
            self.pusher.stream(name, record)

    def metric(self, step: int, **values: float) -> None:
        if self.pusher is not None:
            self.pusher.metrics(step=step, **values)

    def metrics(self, step: int, **values: float) -> None:
        self.metric(step, **values)

    def heartbeat(self, step: int, **values: Any) -> None:
        if self.pusher is not None:
            self.pusher.heartbeat(step=step, **values)

    def sample(self, step: int, prompt: str, completion: str,
               max_chars: int = 600) -> None:
        if self.pusher is not None:
            self.pusher.canary_sample(
                step=step,
                prompt=prompt,
                completion=completion,
                max_chars=max_chars,
            )

    def event(self, type: str, step: int, details: str = "",
              reasoning: str = "", **extra: Any) -> None:
        if self.pusher is not None:
            self.pusher.event(
                type=type,
                step=step,
                details=details,
                reasoning=reasoning,
                **extra,
            )

    def sync_archive(self) -> None:
        if self.archive is not None:
            self.archive.sync_now()

    def flush(self) -> None:
        if self.pusher is not None:
            self.pusher.flush()
        self.sync_archive()

    def complete(
        self,
        *,
        final_state: str = "completed",
        final_metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.pusher is not None:
            self.pusher.complete(
                final_state=final_state,
                final_metrics=final_metrics,
            )
        if self.archive is not None:
            self.archive.complete()
