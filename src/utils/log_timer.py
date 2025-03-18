from contextlib import contextmanager, asynccontextmanager
import time
import logging
import logging_config

logger = logging.getLogger(__name__)


@contextmanager
def log_timer(name):
    """Helper function to log time for functions"""
    start = time.time()
    yield
    elapsed = time.time() - start
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    logger.info(f"{name} took {elapsed_str}.")


@asynccontextmanager
async def async_log_timer(name: str):
    """Helper function to log time for functions - async"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        logger.info(f"{name} took {elapsed_str}.")
