"""
File: preprocess_missing_data_async.py
Author: Xiao-Fei Zhang
Date: 2024 Aug 6

Function file to detect file encoding.

Dependency: chardet
"""

from pathlib import Path
import chardet


def detect_encoding(file_path: Path | str):
    """
    Detects the encoding of a given file.

    Args:
    - file_path (str): The path to the file.

    Returns:
    - encoding (str): The detected encoding of the file.
    - confidence (float): The confidence level of the detected encoding.
    """
    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        file_encoding = result["encoding"]
        encoding_confidence = result["confidence"]
        return file_encoding, encoding_confidence


if __name__ == "__main__":
    # Example usage
    f_path = "your_file_here.txt"  # Replace with your file path
    encoding, confidence = detect_encoding(f_path)
    print(f"Detected encoding: {encoding}, Confidence: {confidence}")
