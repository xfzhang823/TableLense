"""
Helper file to dynamically generate output file names for specific topics
by "piecing" together root, llm provider, indexed or not, topic, etc.
"""

from pathlib import Path
from typing import Optional
import logging
import logging_config


# Setup logger
logger = logging.getLogger(__name__)


def generate_dynamic_file_name(
    file_root: str,
    llm_provider: str,
    index_status: Optional[str] = None,
    topic: Optional[str] = None,
    file_ext: str = "json",
) -> str:
    """
    Dynamically generates a file name based on the file root, LLM provider,
    indexing status, topic, and file extension. All input parameters are sanitized
    to replace spaces with underscores.

    Args:
        file_root (str): The base name of the file (e.g., "array_of_thoughts").
        llm_provider (str): The LLM provider (e.g., "openai", "claude").
        index_status (Optional[str], optional): Specifies if the file is indexed or not
            (e.g., "with_index", "without_index"). Defaults to None.
        topic (Optional[str], optional): The subject or topic of the file
            (e.g., "hardware_management").
            Defaults to None.
        file_ext (str, optional): The file extension (default is "json").

    Returns:
        str: The dynamically generated file name.
    """
    # Sanitize inputs: replace spaces with underscores for all provided parameters
    sanitized_inputs = [
        (param.replace(" ", "_") if param else "")
        for param in [file_root, index_status, topic, llm_provider]
    ]

    # Combine components and filter out empty strings
    components = [comp for comp in sanitized_inputs if comp]

    # Create the final file name
    file_name = "_".join(components) + f".{file_ext.strip()}"

    logger.info(f"File name ({file_name}) created.")

    return file_name
