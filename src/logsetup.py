import logging

def setup_logging(log_file, log_level=logging.INFO, handler=logging.StreamHandler()):
    """Setup logging configuration"""
    format = '%(asctime)s %(message)s'
    datefmt = '%m/%d/%Y %I:%M:%S %p'

    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger

