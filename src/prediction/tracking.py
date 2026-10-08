"""Optional Weights & Biases logging.

Wrapped so train.py carries no `if args.wandb:` branches: when tracking is off,
every method is a no-op. wandb is imported lazily, so it is not a dependency of
this package -- but asking for `--wandb` without it installed is an error rather
than a silent skip, since a run you thought was logged and wasn't is worse than
a crash at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _plain(value: Any) -> Any:
    """Make argparse values JSON/wandb-safe."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class Tracker:
    """Thin wandb wrapper. `Tracker(enabled=False)` is a working no-op."""

    def __init__(self, enabled: bool = False, project: str = "ebv-mil",
                 entity: str | None = None, name: str | None = None,
                 group: str | None = None, config: dict | None = None) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "ERROR: --wandb requires the wandb package (pip install wandb, "
                "then `wandb login`). Drop --wandb to train without tracking."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=project, entity=entity, name=name, group=group,
            config={k: _plain(v) for k, v in (config or {}).items()},
        )

    def log(self, **values) -> None:
        """One row of per-epoch or per-fold metrics."""
        if self._run is not None:
            self._wandb.log(values)

    def summary(self, **values) -> None:
        """Run-level headline numbers, shown as columns in the runs table."""
        if self._run is not None:
            self._run.summary.update(values)

    def save(self, path: Path) -> None:
        """Attach an output file (predictions.csv, report.json) to the run."""
        if self._run is not None:
            self._wandb.save(str(path), policy="now")

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
