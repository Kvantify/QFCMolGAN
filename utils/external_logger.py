from typing import Any, Optional

import wandb
import torch.nn as nn


class WandbLogger(object):
    def __init__(
        self,
        project_name: str,
        config: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> None:
        """Set up the external logger."""
        self.last_step = 0
        wandb.init(project=project_name, config=config, tags=tags)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log the metrics."""
        if step is not None:
            self.last_step = step
        wandb.log(metrics, step=self.last_step)

    def watch_model(self, model: nn.Module, log: str = "all", log_freq: int = 10) -> None:
        """Watch the model."""
        wandb.watch(model, log=log, log_freq=log_freq)  # type: ignore
