import os
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


class WandbArgsUpdate(L.Callback):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        if getattr(trainer.logger.experiment, "nop", False):
            pass  # not in the logging rank
        else:
            trainer.logger.experiment.config.update(self.args)

