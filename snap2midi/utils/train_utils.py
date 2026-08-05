import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from lightning_fabric.loggers.logger import Logger

def pl_logger(logger_name: str="csv", project_name: str="lightning_logs", \
              version: str|int|None=None) -> Logger:
    """
        Returns a logger instance for
        training. Accepted loggers are
        `csv` and `wandb`.

        Args
        -----
            logger_name (str): logger name
            project_name (str): project name
            version (str | int | None): subdirectory under `./logs/<project_name>/`
                to write into. None -- the default -- keeps the historical
                behaviour of allocating a fresh `version_N` per process, which
                is what a one-shot run wants. A job chained across several
                walltimes wants a fixed string instead: each resumed job would
                otherwise open its own `version_N`, and tensorboard renders
                those as separate runs, so a single training curve arrives
                split into as many fragments as there were jobs.

        Returns
        --------
            logger (Logger): Logger instance
    """
    # version is deliberately not passed to WandbLogger: there it means the run
    # id to resume, not a directory, so forwarding it would silently change
    # wandb's resume semantics for every caller.
    logger_dict = {
        "csv": CSVLogger("./logs", name=project_name, version=version),
        "wandb": WandbLogger(name=project_name),
        "tensorboard": TensorBoardLogger(save_dir="./logs", name=project_name, version=version),
    }

    assert logger_name in logger_dict, "[ERROR] Recognized loggers are `csv`, `tensorboard` and `wandb`"
    return logger_dict[logger_name]
