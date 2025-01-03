"""TBA"""

from pathlib import Path
import asyncio
from pipelines.preprocessing_pipeline_async import run_preprocessing_pipeline_async
import logging
import logging_config

# Set up logger
logger = logging.getLogger(__name__)


async def main():
    """
    Orchestrate pipelines
    """
    # Run preprocessing pipeline
    await run_preprocessing_pipeline_async()


if __name__ == "__main__":
    asyncio.run(main())
