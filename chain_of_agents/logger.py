# logger.py
import logging
import wandb

class WandbHandler(logging.Handler):
    """
    Custom logging handler for W&B.

    NOTE: Disabled by default to prevent excessive wandb.log() calls.
    Logging every statement to W&B creates thousands of API calls and can:
    1. Hit rate limits
    2. Fragment data across too many steps
    3. Cause incomplete syncing due to overwhelming the API

    If you need W&B logging, use explicit wandb.log() calls for important
    metrics only, not for every console message.
    """
    def emit(self, record):
        # Disabled to prevent excessive wandb.log() calls
        # Uncomment only if you understand the implications
        pass
        # log_entry = self.format(record)
        # wandb.log({"log": log_entry})

def setup_logger(name="my_logger", enable_wandb=False):
    """
    Set up a logger with console output.

    Args:
        name: Logger name
        enable_wandb: If True, adds WandbHandler (disabled by default to prevent
                      excessive API calls that cause syncing issues)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # Wandb handler (disabled by default)
        if enable_wandb:
            wandb_handler = WandbHandler()
            wandb_handler.setLevel(logging.INFO)
            logger.addHandler(wandb_handler)

    return logger
