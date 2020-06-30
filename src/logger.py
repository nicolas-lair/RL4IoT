import logging
FORMATTER = logging.Formatter("%(asctime)s %(levelname)-8.8s %(name)-5s %(message)-12.12s")

class Logger:
    def __init__(self, name, level, console=True, log_file=True, **kwargs):
        assert isinstance(name, str)
        assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        assert isinstance(console, bool)
        assert isinstance(log_file, bool)
        self.logger = logging.getLogger(name)
        logger.setLevel(level)
        if isinstance(log_file, str):
            fileHandler = logging.FileHandler(f"{log_file}.log")
            fileHandler.setFormatter(FORMATTER)
            logger.addHandler(fileHandler)
        if console:
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(FORMATTER)
            logger.addHandler(consoleHandler)




if __name__ == '__main__':
    logger = create_logger(__name__, logging.DEBUG, console=True, log_file=None)
    logger.debug('Hello')
    logger.info('Hello')
    logger.warning('Hello')
    logger.critical('Hello')
