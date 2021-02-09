import os
import logging
from pprint import pformat

FORMATTER = logging.Formatter("{asctime} {simulation_id} {levelname:5} {lineno:4}:{filename:20} {message}",
                              datefmt="%H:%M:%S",
                              style='{')

rootLogger = logging.getLogger('root')
extra_dict = dict(simulation_id="")


def set_logger_handler(level, console=True, log_file=True, log_path=None, simulation_id=""):
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    assert isinstance(console, bool)
    assert isinstance(log_file, bool)
    global SIMULATION_ID
    SIMULATION_ID = simulation_id
    rootLogger.setLevel(level)
    if log_file:
        assert isinstance(log_path, str)
        add_handler(rootLogger, log_path, type='file')
    if console:
        add_handler(rootLogger, log_path, type='stream')


def add_handler(logger, log_path, type='stream'):
    if type == 'stream':
        handler = logging.StreamHandler()
    elif type == 'file':
        handler = logging.FileHandler(os.path.join(log_path, "log.log"))
    else:
        raise NotImplementedError
    handler.setFormatter(FORMATTER)
    logger.addHandler(handler)


def update_log_file_path(log_path):
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            rootLogger.removeHandler(handler)
    add_handler(rootLogger, log_path, type='file')


def update_logger(log_path, simulation_id=""):
    update_log_file_path(log_path)
    extra_dict.update(dict(simulation_id=simulation_id))


def get_logger(name):
    logger = rootLogger.getChild(name)
    adapter = logging.LoggerAdapter(logger, extra=extra_dict)
    return adapter


def format_oracle_state_log(d):
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
