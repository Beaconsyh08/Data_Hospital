#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2021/11/11 21:26:05
@Author  :   chenminghua 
'''

# import time
import logging
import sys
from os import makedirs
from os.path import dirname, exists

import colorlog
from configs.config import LogConfig

loggers = {}

LOG_ENABLED = True  # 是否开启日志
LOG_TO_CONSOLE = True  # 是否输出到控制台
LOG_TO_FILE = True  # 是否输出到文件
LOG_TO_ES = True  # 是否输出到 Elasticsearch

LOG_PATH = LogConfig.LOG_PATH  # 日志文件路径
LOG_LEVEL = 'DEBUG'  # 日志级别
# LOG_FORMAT = '%(levelname)s - %(asctime)s - process: %(process)d - %(filename)s - %(name)s - %(lineno)d - %(module)s - %(message)s'  # 每条日志输出格式
# LOG_FORMAT = '%(levelname)-10s %(asctime)s: [%(filename)s:%(lineno)d] %(message)s'
ELASTIC_SEARCH_HOST = 'eshost'  # Elasticsearch Host
ELASTIC_SEARCH_PORT = 9200  # Elasticsearch Port
ELASTIC_SEARCH_INDEX = 'runtime'  # Elasticsearch Index Name
APP_ENVIRONMENT = 'dev'  # 运行环境，如测试环境还是生产环境

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
COLOUR_FORMAT = "%(log_color)s%(levelname)-8s%(asctime)s.%(msecs)-4d[%(filename)s:%(lineno)d] %(message)s"
COLOURS = {'DEBUG': 'green',
           'INFO': 'white',
           'WARNING': 'bold_yellow',
           'ERROR': 'bold_red',
           'CRITICAL': 'bold_purple'}


def get_logger(name=None):
    """
    DEBUG
        Detailed information, typically of interest only when diagnosing problems.
    INFO
        Confirmation that things are working as expected.
    WARNING
        An indication that something unexpected happened, or indicative of some problem in the near future 
        (e.g. ‘disk space low’). The software is still working as expected.
    ERROR
        Due to a more serious problem, the software has not been able to perform some function.
    CRITICAL
        A serious error, indicating that the program itself may be unable to continue running.
    """
    global loggers

    if not name: name = __name__

    if loggers.get(name):
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)


    # 输出到控制台
    if LOG_ENABLED and LOG_TO_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=LOG_LEVEL)
        formatter = colorlog.ColoredFormatter(COLOUR_FORMAT, DATE_FORMAT, log_colors=COLOURS)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 输出到文件
    if LOG_ENABLED and LOG_TO_FILE:
        # 如果路径不存在，创建日志文件文件夹
        log_dir = dirname(LOG_PATH)
        if not exists(log_dir): makedirs(log_dir)
        # 添加 FileHandler
        file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
        file_handler.setLevel(level=LOG_LEVEL)
        formatter = colorlog.ColoredFormatter(COLOUR_FORMAT, DATE_FORMAT, log_colors=COLOURS)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 保存到全局 loggers
    loggers[name] = logger
    return logger
    