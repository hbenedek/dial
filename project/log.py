import logging

def init_logger(log_path: str="logs", file_name: str="bug", level: str = "info") -> logging.Logger:
    """Initializes a Logger object"""
    log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)5s] [%(filename)s:%(lineno)s - %(funcName)s()]: %(message)s" )
    logger = logging.getLogger()

    file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logger.addHandler(console_handler)
    logger = set_level(logger, level)
    return logger

def set_level(logger:logging.Logger, level: str) -> logging.Logger:
    """Changes the logging level, two possible options(info/debug)"""
    if level == "info":
        logger.setLevel(logging.INFO)
    if level == "debug":
        logger.setLevel(logging.DEBUG)
    return logger

logger = init_logger()