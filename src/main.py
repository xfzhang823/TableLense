"""TBA"""

import asyncio
from pipelines.preprocessing_pipeline_async import preprocessing_pipeline_async
from project_config import (
    YEARBOOK_2012_DATA_DIR,
    YEARBOOK_2022_DATA_DIR,
    preprocessed_2012_data_file,
)
from nn_models.training_utils import process_batch

import logging
import logging_config

# Set up logger
logger = logging.getLogger(__name__)


async def main():
    """Orchestrate pipelines"""
    # Step 1. Run preprocessing pipeline
    await preprocessing_pipeline_async()

    # Step 2. Run data labeling pipeline


if __name__ == "__main__":
    asyncio.run(main())
