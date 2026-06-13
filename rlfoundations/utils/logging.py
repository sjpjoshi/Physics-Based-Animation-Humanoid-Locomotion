"""Thin Weights & Biases wrapper.

Keeps experiment logging in one place and makes it safe to run without a W&B
account: the default mode is "offline" (writes to ./wandb/, which is gitignored).
Run `wandb login` and pass mode="online" for the hosted dashboard, or "disabled"
to turn logging off entirely.
"""
from __future__ import annotations

from typing import Any


class Logger:
    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        mode: str = "offline",
        run_name: str | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            mode=mode,
            reinit="finish_previous",
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        self.run.finish()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()
