"""Read data and header information from raw file."""

from pathlib import Path

from philips_ismrmrd.utils import _extract_attributes
from philips_ismrmrd.utils import _extract_general_info
from philips_ismrmrd.utils import _preallocate_samples
from philips_ismrmrd.utils import _read_and_store_samples_per_type
from philips_ismrmrd.utils import _remove_empty_fields
from philips_ismrmrd.utils import _split_attributes_per_type
from philips_ismrmrd.utils import _validate_path


def read_data_list(path_to_data_or_list: str, remove_empty_fields: bool = True):
    """Read data and attributes from a .data/.list file pair."""
    # Remove extension from path if any
    path = str(Path(path_to_data_or_list).parent / Path(path_to_data_or_list).stem)

    # Read the .list file to get attributes and general info
    attributes, info = _read_list_file(f'{path}.list')

    # Preallocate arrays for samples per type
    samples_per_type = _preallocate_samples(attributes)

    # Read and store samples from .data file
    _read_and_store_samples_per_type(samples_per_type, f'{path}.data', attributes)

    # Split attributes DataFrame into separate DataFrames per type
    attributes_per_type = _split_attributes_per_type(attributes)

    # Optionally remove empty fields
    if remove_empty_fields:
        samples_per_type, attributes_per_type = _remove_empty_fields(samples_per_type, attributes_per_type)

    return samples_per_type, attributes_per_type, info


def _read_list_file(path_to_list_file: str):
    # Validate that the file exists, is non-empty, and has .list extension
    _validate_path(path_to_list_file, '.list')

    # Read all lines from the .list file
    with Path(path_to_list_file).open() as f:
        list_lines = f.readlines()

    list_lines = [line.rstrip('\n') for line in list_lines]

    # Extract general scan info lines (starting with '#' or '.')
    general_info = _extract_general_info(list_lines)

    # Extract attributes of complex data vectors as DataFrame
    attributes = _extract_attributes(list_lines)

    return attributes, general_info
