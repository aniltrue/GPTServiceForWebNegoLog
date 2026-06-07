import asyncio
import atexit
import logging
import os
import queue
import sys
import threading
from typing import Any, Tuple, List, Optional

_LogItem = Tuple[int, str, tuple, dict]


class NoExcInfoFormatter(logging.Formatter):
    """Formatter that omits exception traceback — used for console and info-log handlers."""

    def format(self, record: logging.LogRecord) -> str:
        exc_info = record.exc_info
        exc_text = record.exc_text
        record.exc_info = None
        record.exc_text = None
        result = super().format(record)
        record.exc_info = exc_info
        record.exc_text = exc_text
        return result


class AsyncLogger:
    def __init__(
            self,
            name: str,
            level: int = logging.INFO,
            *,
            handlers: Optional[List[logging.Handler]] = None,
            propagate: bool = False,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = propagate

        if handlers:
            for h in handlers:
                if h not in self.logger.handlers:
                    self.logger.addHandler(h)

        self.queue: queue.Queue[Optional[_LogItem]] = queue.Queue()
        self.stopped = threading.Event()
        self.started = False
        self.logged = threading.Event()
        self.lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None

    async def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()
        atexit.register(self._atexit_flush)

    async def close(self) -> None:
        if not self.started or self.stopped.is_set():
            return
        self.stopped.set()
        with self.lock:
            self.queue.put(None)
        if self.worker:
            self.worker.join()
        self.worker = None
        self.started = False

    def _run_worker(self) -> None:
        while not self.stopped.is_set():
            self.logged.wait(5)
            if not self.logged.is_set():
                continue
            with self.lock:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    self.logged.clear()
                    continue
                level, msg, args, kwargs = item
                self.logger.log(level, msg, *args, **kwargs)
                self.queue.task_done()
                self.logged.clear()

    def _atexit_flush(self):
        asyncio.run(self.close())

    def __enter__(self):
        asyncio.run(self.start())
        return self

    def __exit__(self, exc_type, exc, tb):
        asyncio.run(self.close())

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        with self.lock:
            self.queue.put((level, msg, args, kwargs))
            self.logged.set()

    async def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        await asyncio.to_thread(self.log, logging.DEBUG, msg, *args, **kwargs)

    async def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        await asyncio.to_thread(self.log, logging.INFO, msg, *args, **kwargs)

    async def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        await asyncio.to_thread(self.log, logging.WARNING, msg, *args, **kwargs)

    async def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if "exc_info" not in kwargs or kwargs.get("exc_info") is True:
            exc = sys.exc_info()
            kwargs["exc_info"] = exc if exc[0] is not None else None
        await asyncio.to_thread(self.log, logging.ERROR, msg, *args, **kwargs)

    async def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs["exc_info"] = sys.exc_info()
        await asyncio.to_thread(self.log, logging.ERROR, msg, *args, **kwargs)


def get_logger(service_name: str, logging_level: int = logging.INFO) -> AsyncLogger:
    directory = "logs"
    os.makedirs(directory, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    plain_formatter = NoExcInfoFormatter(fmt)   # message only, no traceback
    full_formatter = logging.Formatter(fmt)      # with traceback

    # Console: message only, no traceback
    console = logging.StreamHandler()
    console.setFormatter(plain_formatter)

    # INFO/WARNING -> info.log, no traceback
    info_handler = logging.FileHandler(f"{directory}/{service_name}_info.log")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda r: r.levelno <= logging.WARNING)
    info_handler.setFormatter(plain_formatter)

    # ERROR -> error.log with full traceback
    error_handler = logging.FileHandler(f"{directory}/{service_name}_error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(full_formatter)

    return AsyncLogger(
        service_name,
        logging_level,
        handlers=[console, info_handler, error_handler],
        propagate=False
    )
