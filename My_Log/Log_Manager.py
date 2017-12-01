import logging
import logging.handlers

logger = None
file_logger = None
# def define():
log_file = "processing.log"
file_handle = logging.FileHandler(log_file)
con_handle = logging.StreamHandler()
fmt = "%(asctime)s:\n  %(message)s"
log_fmt = logging.Formatter(fmt)
file_handle.setFormatter(log_fmt)
con_handle.setFormatter(log_fmt)
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
logger.addHandler(file_handle)
logger.addHandler(con_handle)
file_logger = logging.getLogger("file_logger")
file_logger.setLevel(logging.INFO)
file_logger.addHandler(file_handle)

def print_info(info):
    logger.info(info)

def print_info_file(info):
    file_logger.info(info)
