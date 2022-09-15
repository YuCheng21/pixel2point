import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
# the logger with handler will use higher one level (!important)
logger.setLevel(logging.DEBUG)


def console_logger():
    console_handler = logging.StreamHandler(sys.stderr)
    console_format = logging.Formatter(
        fmt='%(asctime)s: %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)


def file_logger():
    log_path = Path('./logs/app.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_path, maxBytes=1 * 10 ** 6, backupCount=10, encoding='UTF-8', delay=False)
    file_format = logging.Formatter(
        fmt='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
