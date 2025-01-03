"""
File: preprocess_missing_data_async.py
Author: Xiao-Fei Zhang
Date: 2024 Aug 6

Function file to detect file encoding.

Dependency: chardet
"""

import chardet


def detect_encoding(file_path):
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
        encoding = result["encoding"]
        confidence = result["confidence"]
        return encoding, confidence


if __name__ == "__main__":
    # Example usage
    f_path = "your_file_here.txt"  # Replace with your file path
    encoding, confidence = detect_encoding(f_path)
    print(f"Detected encoding: {encoding}, Confidence: {confidence}")
