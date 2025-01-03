import logging
import logging.handlers
import os
import getpass
from datetime import datetime
from utils.find_project_root import find_project_root


class EnhancedLoggerSetup:
    """Class for custom logger."""

    def __init__(self, app_name=None):
        """
        Initialize logging setup with application name and logging directories

        Args:
            app_name (str, optional): Name of your application
        """
        self.app_name = app_name or "app"
        self.root_dir = find_project_root()
        self.logs_dir = os.path.join(self.root_dir, "logs")
        self.username = self.get_username()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique session ID

        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize formatters
        self.detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        self.console_formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        )

    def get_username(self):
        """
        Retrieves the current username.

        Returns:
            str: Username of the current user.
        """
        username = getpass.getuser()
        print(f"[DEBUG] Retrieved username: {username}")
        return username

    def get_log_file_path(self, log_type):
        """
        Constructs the log file path based on the username, session ID, and log type.

        Args:
            log_type (str): Type of log file (error, debug, info)

        Returns:
            str: Log file path
        """
        filename = f"{self.username}_{self.session_id}_{log_type}.log"
        log_path = os.path.join(self.logs_dir, filename)
        print(f"[DEBUG] Log file path for {log_type}: {log_path}")
        return log_path

    def setup_file_handler(self, log_type, level):
        """
        Sets up a rotating file handler for the specified log type.

        Args:
            log_type (str): Type of log file
            level (int): Logging level for this handler

        Returns:
            RotatingFileHandler: Configured file handler
        """
        log_path = self.get_log_file_path(log_type)
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        handler.setLevel(level)
        handler.setFormatter(self.detailed_formatter)
        return handler

    def setup_console_handler(self):
        """
        Sets up a console handler for logging.

        Returns:
            StreamHandler: Configured console handler
        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        return console_handler

    def configure_logging(self):
        """
        Configures logging settings with multiple handlers and formatters.
        """
        try:
            # Get the root logger
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)

            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Set up handlers for different log levels
            error_handler = self.setup_file_handler("error", logging.ERROR)
            debug_handler = self.setup_file_handler("debug", logging.DEBUG)
            info_handler = self.setup_file_handler("info", logging.INFO)
            console_handler = self.setup_console_handler()

            # Add all handlers
            logger.addHandler(error_handler)
            logger.addHandler(debug_handler)
            logger.addHandler(info_handler)
            logger.addHandler(console_handler)

            # Prevent duplicate logs
            logger.propagate = False

            print(f"[DEBUG] Logging successfully configured with multiple handlers")

        except Exception as e:
            print(f"Failed to configure logging: {e}")
            raise


def configure_logging(app_name=None):
    """
    Main function to configure logging system.

    Args:
        app_name (str, optional): Name of the application
    """
    logger_setup = EnhancedLoggerSetup(app_name)
    logger_setup.configure_logging()


# Automatically configure logging when this module is imported
if __name__ != "__main__":
    configure_logging()
