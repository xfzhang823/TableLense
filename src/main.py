"""
main.py
Orchestrate all the pipelines.
"""

from pathlib import Path
import asyncio
from transformers import AutoModel, AutoTokenizer
from pipelines.preprocessing_pipeline_async import run_preprocessing_pipeline_async
from pipelines.data_labeling_pipeline import run_data_labeling_pipeline
from pipelines.model_train_pipeline import run_model_training_pipeline
from pipelines.inference_pipeline_async_batched import (
    run_inference_pipeline_async_batched,
)
from pipelines.tables_reconstruction_pipeline_async import (
    run_tables_reconstruction_pipeline_async,
)

import logging
import logging_config

# Set up logger
logger = logging.getLogger(__name__)

# # Load the models
# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


async def main():
    """
    Orchestrate pipelines
    """
    # Run preprocessing pipeline
    await run_preprocessing_pipeline_async()

    # Run data labeling pipeline (sync function → move to separate thread)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, run_data_labeling_pipeline
    )  # passing a function reference

    # Run training pipeline (sync function → move to separate thread)
    await loop.run_in_executor(
        None, run_model_training_pipeline
    )  # passing a function reference

    # Run inference pipeline()
    await run_inference_pipeline_async_batched()

    # Run inference pipeline()
    await run_tables_reconstruction_pipeline_async()


if __name__ == "__main__":
    asyncio.run(main())
