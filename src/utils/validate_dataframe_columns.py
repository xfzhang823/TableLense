"""Helper function to validate df columns required"""

import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    optional_cols: List[str] = None,
    strict: bool = True,
) -> None:
    """
    Validate that the given DataFrame contains the required columns.
    Optionally check for optional columns and decide if missing them should
    raise an error or just log a warning.

    Args:
        df (pd.DataFrame):
            The DataFrame to validate.
        required_cols (List[str]):
            Columns that must be present in the DataFrame. If any are missing,
            the function raises a ValueError (if strict=True).
        optional_cols (List[str], optional):
            Columns that are recommended but not strictly required.
            If any are missing, we log a warning. Defaults to None.
        strict (bool, optional):
            If True, missing required columns raises a ValueError.
            If False, logs a warning instead of raising. Defaults to True.

    Raises:
        ValueError:
            If any required columns are missing and strict=True.

    Logs:
        An info message "Validation passed: All required columns are present."
        if no required columns are missing.
    """
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        msg = (
            f"Missing required columns: {missing_required}. "
            f"Available columns are: {list(df.columns)}"
        )
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    if optional_cols:
        missing_optional = [col for col in optional_cols if col not in df.columns]
        if missing_optional:
            logger.warning(
                f"Missing optional columns: {missing_optional}. "
                f"Available columns are: {list(df.columns)}"
            )

    logger.info("Validation passed: All required columns are present.")
