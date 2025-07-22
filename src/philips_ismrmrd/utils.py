import os
import pandas as pd
import numpy as np
import re
from io import StringIO

# Raw data from Philips MR system stored as single precision complex floats
COMPLEX_ELTYPE = np.complex64

# Types of "complex data vectors" in the .list file
COMPLEX_DATA_VECTOR_TYPES = ["STD", "REJ", "PHX", "FRX", "NOI", "NAV", "DNA"]

# Dimensions of the data vectors in the .list file
DIMENSIONS_STD = (
    "kx", "ky", "kz", "loca", "chan", "aver", "dyn", "mix", "card",
    "echo", "extr1", "extr2", "rf", "grad"
)

def _validate_path(path_to_file: str, expected_extension: str):
    ext = os.path.splitext(path_to_file)[1]
    if expected_extension.startswith("."):
        expected_extension = expected_extension[1:]
    if ext != f".{expected_extension}":
        raise ValueError(f"Invalid file extension: expected '.{expected_extension}', got '{ext}'")
    if os.path.getsize(path_to_file) == 0:
        raise ValueError(f"File is empty: {path_to_file}")

def _extract_general_info(list_lines):
    general_info = [line for line in list_lines if line.startswith("#") or line.startswith(".")]
    if not general_info:
        raise ValueError("No general information found in .list file")
    return general_info

def _get_attributes_header(list_lines):
    index = next((i for i, line in enumerate(list_lines) if "START OF DATA VECTOR INDEX" in line), None)
    if index is None:
        raise ValueError("Could not find 'START OF DATA VECTOR INDEX' in .list file")
    header_line = list_lines[index + 2]
    header_line = re.sub(r"#\s+", "", header_line.strip())
    header_line = re.sub(r"\s+", ",", header_line)
    return header_line.split(",")

def _extract_attributes(list_lines):
    attribute_lines = [line for line in list_lines if not (line.startswith("#") or line.startswith("."))]
    if not attribute_lines:
        raise ValueError("No attributes found in .list file")
    attribute_lines = [re.sub(r"\s+", ",", line.strip()) for line in attribute_lines]
    attributes_str = "\n".join(attribute_lines)
    header = _get_attributes_header(list_lines)
    df = pd.read_csv(StringIO(attributes_str), names=header)
    return df

def _split_attributes_per_type(attributes: pd.DataFrame):
    return {typ: attributes[attributes["typ"] == typ] for typ in COMPLEX_DATA_VECTOR_TYPES}

def _num_bytes_to_num_samples(bytes_: int):
    elsize = COMPLEX_ELTYPE().nbytes
    if bytes_ % elsize != 0:
        raise ValueError("bytes is not divisible by sizeof(COMPLEX_ELTYPE)")
    return bytes_ // elsize

def _total_num_samples(type_, attributes: pd.DataFrame):
    total_bytes = attributes[attributes["typ"] == str(type_)]["size"].sum()
    return _num_bytes_to_num_samples(total_bytes)

def _preallocate_samples(attributes: pd.DataFrame):
    samples = {}
    for typ in COMPLEX_DATA_VECTOR_TYPES:
        total = _total_num_samples(typ, attributes)
        samples[typ] = np.empty(total, dtype=COMPLEX_ELTYPE)
    return samples

def _read_and_store_samples_per_type(samples: dict, path_to_datafile: str, attributes: pd.DataFrame):
    with open(path_to_datafile, "rb") as f:
        offset_tracker = {typ: 0 for typ in COMPLEX_DATA_VECTOR_TYPES}
        for _, row in attributes.iterrows():
            offset = int(row["offset"])
            size = int(row["size"])
            f.seek(offset)
            raw = f.read(size)
            num_samples = _num_bytes_to_num_samples(size)
            data = np.frombuffer(raw, dtype=COMPLEX_ELTYPE)
            typ = row["typ"]
            start = offset_tracker[typ]
            end = start + num_samples
            samples[typ][start:end] = data
            offset_tracker[typ] = end
        if f.read(1):
            print("Warning: did not reach end of file")

def _offset_and_size_to_range(offset: int, size: int):
    start = _num_bytes_to_num_samples(offset)
    stop = start + _num_bytes_to_num_samples(size)
    return range(start, stop)

def _remove_empty_fields(samples: dict, attributes: dict):
    common_keys = [k for k in samples if len(samples[k]) > 0 and not attributes[k].empty]
    return {k: samples[k] for k in common_keys}, {k: attributes[k] for k in common_keys}
