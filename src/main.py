"""TBA"""

from pathlib import Path
import asyncio
from pipelines.preprocessing_pipeline_async import run_preprocessing_pipeline_async
from pipelines.data_labeling_pipeline import run_data_labeling_pipeline
import logging
import logging_config

# Set up logger
logger = logging.getLogger(__name__)


async def main():
    """
    Orchestrate pipelines
    """
    # Run preprocessing pipeline
    # await run_preprocessing_pipeline_async()

    # Run data labeling pipeline
    run_data_labeling_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
