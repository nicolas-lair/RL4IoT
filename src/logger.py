import os
import logging
from pprint import pformat
import yaml
from config import params

FORMATTER = logging.Formatter("{asctime} {levelname:5} {lineno:4}:{filename:20} {message}", style='{')


def create_logger(name, level, console=True, log_file=True, log_path=None):
    assert isinstance(name, str)
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    assert isinstance(console, bool)
    assert isinstance(log_file, bool)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if log_file:
        assert isinstance(log_path, str)
        fileHandler = logging.FileHandler(os.path.join(log_path, "log.log"))
        fileHandler.setFormatter(FORMATTER)
        logger.addHandler(fileHandler)
    if console:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(FORMATTER)
        logger.addHandler(consoleHandler)
    return logger

def format_user_state_log(d):
    def aux(d):
        for v in d.values():
            if isinstance(v, dict):
                try:
                    v.pop('embedding')
                except KeyError:
                    pass
                aux(v)

    u = d.copy()
    aux(u)
    return pformat(u)

rootLogger = create_logger(name='root', **params['logger'], log_path=params['save_directory'])

# rootLogger.info('hello')
