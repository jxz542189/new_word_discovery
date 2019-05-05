# coding:utf-8
import logging.handlers
import os


path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class Logger(object):
    log_info = logging.getLogger('I')
    log_info.setLevel(logging.INFO)

    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(path, 'log', 'info_log.txt'), when='D', interval=30, backupCount=30,
                                                        delay=False, utc=False)
    handler.suffix = "%Y%m%d_%H%M.log"
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    log_info.addHandler(handler)
    log_info.addHandler(stream_handler)

    log_error = logging.getLogger('E')
    log_error.setLevel(logging.ERROR)
    handler_error = logging.handlers.TimedRotatingFileHandler(os.path.join(path, 'log', 'error_log.txt'), when='D', interval=180, backupCount=30,
                                                        delay=False, utc=False)

    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    handler_error.setFormatter(formatter)

    log_error.addHandler(handler_error)
    log_error.addHandler(stream_handler)


if __name__ == "__main__":
    Logger.log_info.info('谢谢')
    Logger.log_error.error('出错了')

