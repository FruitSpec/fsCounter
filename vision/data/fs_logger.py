import os
import logging
from functools import wraps
import time


class Logger():

    def __init__(self, args, handler=None, name='fscounter'):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.set_handler(args, handler)

        self.statistics = []
        self.iterations = 0

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def exception(self, msg):
        self.logger.exception(msg)

    def set_handler(self, args, handler=None):
        if handler is None:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler = logging.FileHandler(os.path.join(args.output_folder, 'pipeline.log'), 'a+')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.handler = handler
        else:
            self.handler = handler

    def get_handler(self):
        return self.handler




def log_func(logger):

    # logger is the logging object
    # exception is the decorator objects
    # that logs every exception into log file
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            try:
                s = time.time()
                logger.info(f"{func.__name__} strated")
                output = func(*args, **kwargs)
                logger.info(f"{func.__name__} ended")
                e = time.time()
                logger.info(f"Execution time {e-s:.2f}")

                return output

            except:
                logger.exception("Exception occurred")
            raise

        return wrapper

    return decorator



