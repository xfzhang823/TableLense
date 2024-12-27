"""
File: sklearn_pipeline.py
Author: Xiao-Fei Zhang
Date: last updated on 

Description: Scikit-learn pipeline for preprocessing, training, and inference.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from preprocessing.preprocess_data_sync import preprocess_data_sync


def build_preprocessing_pipeline():
    pipeline = Pipeline(
        [
            ("preprocess", preprocess_data_sync),  # Custom preprocessing function
            ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
            ("scaler", StandardScaler()),  # Scale features
        ]
    )
    return pipeline


def preprocess_and_save_data(raw_data_path, processed_data_path):
    # Load raw data
    raw_data = pd.read_csv(raw_data_path)

    # Build the pipeline
    pipeline = build_preprocessing_pipeline()

    # Preprocess data
    processed_data = pipeline.fit_transform(raw_data)

    # Save processed data
    pd.DataFrame(processed_data).to_csv(processed_data_path, index=False)
