import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class WandbArgsUpdate(pl.Callback):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        if getattr(trainer.logger.experiment, "nop", False):
            pass  # not in the logging rank
        else:
            trainer.logger.experiment.config.update(self.args)


def get_checkpoint_callback(args):
    if args.test_only:
        checkpoint_callback = None
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="Validation Average Loss",
            dirpath=os.path.join(args.checkpoint, args.name),
            filename=args.name,
            mode="min",
        )
    return checkpoint_callback
