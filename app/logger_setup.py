import logging
from app.config.config import Config

def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(Config.get_log_level())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(Config.get_log_level())
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger