"""
Module for configuring application logging.

Provides functions to set up logging handlers, formatters, and file rotation.

Version: 1.0
Author: Xiao-Fei Zhang

Example:
    >>> import logging
    >>> import logging_config

    >>> logger = logging.getLogger(__name__)

    >>> logger.debug("Debug message")
    >>> logger.info("Info message")
    >>> logger.warning("Warning message")
    >>> logger.error("Error message")
    >>> logger.critical("Critical message")
"""

import logging
import logging.handlers
import os
import getpass  # getpass.getuser() instead of os.getlogin()
from utils.find_project_root import find_project_root


def get_username():
    """
    Retrieves the current username.

    Returns:
        str: Username of the current user.
    """
    username = getpass.getuser()
    print(f"[DEBUG] Retrieved username: {username}")  # Debugging output
    return username


def get_log_file_path(logs_dir):
    """
    Constructs the log file path based on the username.

    Args:
        logs_dir (str): Directory path for logs.

    Returns:
        str: Log file path with username.
    """
    username = get_username()
    log_file_path = os.path.join(logs_dir, f"{username}_app.log")
    print(f"[DEBUG] Log file path: {log_file_path}")  # Debugging output
    return log_file_path


def configure_logging():
    """
    Configures logging settings, including handlers, formatters, and file rotation.

    Returns:
        None
    """
    try:
        # Ensure the logs directory exists
        root_dir = find_project_root()
        logs_dir = os.path.join(root_dir, "logs")
        print(
            f"[DEBUG] Root directory: {root_dir}, Logs directory: {logs_dir}"
        )  # Debugging output

        os.makedirs(logs_dir, exist_ok=True)
        print(f"[DEBUG] Logs directory created or already exists.")  # Debugging output

        # Set up log file path
        log_file_path = get_log_file_path(logs_dir)

        # Set up log file rotation: max 10MB per file, up to 5 backup files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        print(f"[DEBUG] File handler configured.")  # Debugging output

        # Create a console handler with a specific log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        print(f"[DEBUG] Console handler configured.")  # Debugging output

        # Get the root logger and configure it
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Set the root logger level
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        print(
            f"[DEBUG] Handlers attached to root logger: {logger.handlers}"
        )  # Debugging output

        # Prevent duplicate logs by disabling propagation for sub-loggers
        logger.propagate = False

        print(f"[DEBUG] Logging successfully configured.")  # Debugging output

    except Exception as e:
        print(f"Failed to configure logging: {e}")
        raise


# Automatically configure logging when this module is imported
configure_logging()
