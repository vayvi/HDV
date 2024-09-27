# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import json
import logging
import os
import sys
import time
import traceback
from typing import Optional, Iterable

import numpy as np
from termcolor import colored
from tqdm import tqdm


def pprint(o):
    if isinstance(o, str):
        try:
            return json.dumps(json.loads(o), indent=4, sort_keys=True)
        except ValueError:
            return o
    elif isinstance(o, dict) or isinstance(o, list) or isinstance(o, tuple):
        if isinstance(o, list):
            o = list(o)
        try:
            return json.dumps(o, indent=4, sort_keys=True)
        except TypeError:
            dict_str = ""
            if isinstance(o, list):
                for v in o:
                    dict_str += f"{v}\n"
            if isinstance(o, dict):
                for k, v in o.items():
                    dict_str += f"{k}:\n{v}\n"
            return dict_str
    else:
        return str(o)


def ppprint(o):
    return "\n".join(pprint(obj) for obj in o)


def fstr(msg = None, color="bold", e: Optional[Exception] = None, w_time=None):
    f_msg = ""
    if w_time:
        f_msg += f"\n\n[{get_time()}]"
    if msg is not None:
        if not isinstance(msg, tuple) and not isinstance(msg, list):
            msg = (msg,)
        f_msg += f"\n\n{get_color(color)}{ppprint(msg)}{ConsoleColors.end}\n"
    if e:
        f_msg += f"\n\n[{e.__class__.__name__}] {e}"
        f_msg += f"\nStack Trace:\n{get_color('red')}{traceback.format_exc()}{ConsoleColors.end}\n"
    return f_msg


def fprint(msg = None, color="bold", e: Optional[Exception] = None, w_time=None):
    print(fstr(msg, color, e, w_time))


class ConsoleColors:
    """
    Last digit
    0	black
    1	red
    2	green
    3	yellow
    4	blue
    5	magenta
    6	cyan
    7	white
    """

    black = "\033[90m"
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"
    white = "\033[97m"
    bold = "\033[1m"
    underline = "\033[4m"
    end = "\033[0m"


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def get_color(color=None):
    return getattr(ConsoleColors, color, "\033[94m")


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="imagenet", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name:
        color:
        distributed_rank:

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d]: %(message)s',
        datefmt='%m/%d %H:%M:%S'
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s.%(msecs)03d]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        output = str(output)
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


base_logger = logging.getLogger("HDV")


class TqdmProgress(tqdm):
    # __init__ that simply extracts a vc_callback from the kwargs
    def __init__(self, iterable, *args, **kwargs):
        self._vc_progress_callback = kwargs.pop("vc_progress_callback", None)
        self._vc_close_callback = kwargs.pop("vc_end_callback", None)
        super().__init__(iterable, *args, **kwargs)

    # Overwrite update to call the callback
    def update(self, n=1):
        displayed = super().update(n)
        if displayed and self._vc_progress_callback is not None:
            self._vc_progress_callback(self)

    # Overwrite close to call the callback
    def close(self):
        super().close()
        if self._vc_close_callback is not None:
            self._vc_close_callback(self)


class SLogger:
    def __init__(self, name, log_file=None):
        self.logger = setup_logger(
            output=log_file,
            name=name,
        )
        np.set_printoptions(precision=3, suppress=True)

    @staticmethod
    def f_str(msg = None, color="bold", e: Optional[Exception] = None):
        # f_msg = ""
        # if msg:
        #     if isinstance(msg, str):
        #         msg = (msg,)
        #     # f_msg += f"\n\n[{get_time()}]"
        #     f_msg += f"\n\n{get_color(color)}{ppprint(msg)}{ConsoleColors.end}\n"
        # if e:
        #     f_msg += f"\n\n[{e.__class__.__name__}] {e}"
        #     f_msg += f"\nStack Trace:\n{get_color('red')}{traceback.format_exc()}{ConsoleColors.end}\n"
        # return f_msg
        return fstr(msg, color, e)

    def info(self, *s, **kwargs) -> None:
        self.logger.info(SLogger.f_str(s, **kwargs))

    def warning(self, *s, exception: Optional[Exception] = None, **kwargs) -> None:
        self.logger.warning(SLogger.f_str(s, color="yellow", e=exception))

    def error(self, *s, exception: Optional[Exception] = None, **kwargs):
        self.logger.warning(SLogger.f_str(s, color="magenta", e=exception))

    def progress(self, current=0, total=None, title="", **kwargs) -> None:
        self.logger.info(f"Progress {title} {current}/{total}")

    def iterate(
        self,
        iterable: Iterable,
        title: str = "",
        total: Optional[int] = None,
        rate_limit: float = 1.0,
    ) -> TqdmProgress:
        self.progress(0, total, title=title)

        def progress_callback(prog: TqdmProgress):
            self.progress(prog.n, prog.total, title)

        def end_callback(prog: TqdmProgress):
            self.logger.info(f"End {title}")

        return TqdmProgress(
            iterable,
            vc_progress_callback=progress_callback,
            vc_end_callback=end_callback,
            desc=title,
            mininterval=rate_limit,
            total=total,
        )

