# logger.py
import logging
import wandb

class WandbHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        wandb.log({"log": log_entry})

def setup_logger(name="my_logger", enable_wandb=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # Wandb handler (only if enabled)
        if enable_wandb:
            wandb_handler = WandbHandler()
            wandb_handler.setLevel(logging.INFO)
            logger.addHandler(wandb_handler)

    return logger
