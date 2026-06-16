import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from lightning_fabric.loggers.logger import Logger

def pl_logger(logger_name: str="csv", project_name: str="lightning_logs") -> Logger:
    """ 
        Returns a logger instance for
        training. Accepted loggers are
        `csv` and `wandb`.

        Args
        -----
            logger_name (str): logger name
            project_name (str): project name
        
        Returns
        --------
            logger (Logger): Logger instance
    """
    logger_dict = {
        "csv": CSVLogger(".", name=project_name),
        "wandb": WandbLogger(name=project_name),
        "tensorboard": TensorBoardLogger(name=project_name),
    }

    assert logger_name in logger_dict, "[ERROR] Recognized loggers are `csv` and `wandb`"
    return logger_dict[logger_name]
