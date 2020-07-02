import os
import logging
from pprint import pformat

FORMATTER = logging.Formatter("{asctime} {levelname:5} {lineno:4}:{filename:20} {message}", style='{')

rootLogger = logging.getLogger('root')


def set_logger_handler(logger, level, console=True, log_file=True, log_path=None):
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    assert isinstance(console, bool)
    assert isinstance(log_file, bool)
    logger.setLevel(level)
    if log_file:
        assert isinstance(log_path, str)
        add_handler(logger, log_path, type='file')
    if console:
        add_handler(logger, log_path, type='stream')
    return logger


def add_handler(logger, log_path, type='stream'):
    if type == 'stream':
        handler = logging.StreamHandler()
    elif type == 'file':
        handler = logging.FileHandler(os.path.join(log_path, "log.log"))
    else:
        raise NotImplementedError
    handler.setFormatter(FORMATTER)
    logger.addHandler(handler)


def update_log_file_path(logger, log_path):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    add_handler(logger, log_path, type='file')


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

# rootLogger.info('hello')
