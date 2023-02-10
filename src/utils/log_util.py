import os
import logging


# Function to return a custom logger
def create_logger(logger_name: str, log_file: str, logger_level=logging.DEBUG,
                  log_format: str = '%(asctime)s:%(name)s:%(message)s'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
