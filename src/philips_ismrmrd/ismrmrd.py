"""Convert raw data to ISMRMRD format."""

import re
import warnings
from pathlib import Path

import ismrmrd
import numpy as np

from philips_ismrmrd.kdata import _calculate_kspace_dimension_ranges
from philips_ismrmrd.kdata import read_data_list


def _extract_single_parameter(general_info, parameter_name):
    """Extract a single parameter from the general info."""
    for line in general_info:
        if parameter_name in line:
            match = re.search(r':\s*(\d+)', line)
            if match:
                return match.group(1)
    raise ValueError(f"Parameter '{parameter_name}' not found in general info.")


def _extract_parameter_pair(general_info, parameter_name):
    """Extract a parameter pair from the general info."""
    for line in general_info:
        if parameter_name in line:
            match = re.search(r':\s*(-?\d+)\s+(-?\d+)', line)
            if match:
                return match.group(1), match.group(2)
    raise ValueError(f"Parameter '{parameter_name}' not found in general info.")


def philips_to_ismrmrd(fname: Path) -> None:
    """Convert Philips raw data in data/list format to ISMRMRD format.

    Both the .data and .list files must be present in the same directory.
    The output will be saved as a .mrd file in the same directory.

    Parameters
    ----------
    fname
        Filename of either data or list data file. Both files must be in the same directory.
    """
    fname_without_extension = Path(fname).parent / Path(fname).stem
    if not fname_without_extension.with_suffix('.data').exists():
        raise ValueError(f'File {fname_without_extension}.data does not exist.')
    if not fname_without_extension.with_suffix('.list').exists():
        raise ValueError(f'File {fname_without_extension}.list does not exist.')
    if fname_without_extension.with_suffix('.mrd').exists():
        raise ValueError(f'File {fname_without_extension}.mrd already exists, please delete it.')
    samples_per_type, attributes_per_type, general_info = read_data_list(str(fname_without_extension))
    samples_to_ismrmrd(str(fname_without_extension) + '.mrd', samples_per_type, attributes_per_type, general_info)


def samples_to_ismrmrd(fname, samples_per_type, attributes_per_type, general_info):
    """Save samples as ISMRMRD file."""
    # Open the dataset
    dataset = ismrmrd.Dataset(fname, 'dataset', create_if_needed=True)

    # Create the XML header and write it to the file
    header = ismrmrd.xsd.ismrmrdHeader()

    # Experimental Conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = 128000000
    warnings.warn('H1resonanceFrequency_Hz is hardcoded to 3T, please change it if needed.', stacklevel=2)
    header.experimentalConditions = exp

    # Acquisition System Information
    sys = ismrmrd.xsd.acquisitionSystemInformationType()
    sys.receiverChannels = int(_extract_single_parameter(general_info, 'number of coil channels'))
    sys.systemVendor = 'Philips'
    header.acquisitionSystemInformation = sys

    # Encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType('cartesian')

    freq_encoding_range = _extract_parameter_pair(general_info, 'kx_range')
    n_freq_encoding = int(float(freq_encoding_range[1]) - float(freq_encoding_range[0]) + 1)
    phase_encoding_1_range = _extract_parameter_pair(general_info, 'ky_range')
    n_phase_encoding_1 = int(float(phase_encoding_1_range[1]) - float(phase_encoding_1_range[0]) + 1)
    if int(_extract_single_parameter(general_info, 'number_of_encoding_dimensions')) == 3:
        phase_encoding_2_range = _extract_parameter_pair(general_info, 'kz_range')
        n_phase_encoding_2 = int(float(phase_encoding_2_range[1]) - float(phase_encoding_2_range[0]) + 1)
    else:
        n_phase_encoding_2 = 1

    # Encoded and recon spaces
    encoding_matrix = ismrmrd.xsd.matrixSizeType()
    encoding_matrix.x = n_freq_encoding
    encoding_matrix.y = n_phase_encoding_1
    encoding_matrix.z = n_phase_encoding_2

    encoding_fov = ismrmrd.xsd.fieldOfViewMm()
    encoding_fov.x = encoding_matrix.x
    encoding_fov.y = encoding_matrix.y
    encoding_fov.z = encoding_matrix.z
    warnings.warn('Encoding resolution of 1mm isotropic is hardcoded, please change it if needed.', stacklevel=2)

    encoding_space = ismrmrd.xsd.encodingSpaceType()
    encoding_space.matrixSize = encoding_matrix
    encoding_space.fieldOfView_mm = encoding_fov
    encoding.encodedSpace = encoding_space

    recon_matrix = ismrmrd.xsd.matrixSizeType()
    recon_matrix.x = int(_extract_single_parameter(general_info, 'X-resolution '))
    recon_matrix.y = int(_extract_single_parameter(general_info, 'Y-resolution '))
    recon_matrix.z = encoding_matrix.z

    recon_fov = ismrmrd.xsd.fieldOfViewMm()
    recon_fov.x = recon_matrix.x
    recon_fov.y = recon_matrix.y
    recon_fov.z = recon_matrix.z
    warnings.warn('Recon resolution of 1mm isotropic is hardcoded, please change it if needed.', stacklevel=2)

    recon_space = ismrmrd.xsd.encodingSpaceType()
    recon_space.matrixSize = recon_matrix
    recon_space.fieldOfView_mm = recon_fov
    encoding.reconSpace = recon_space

    # Helper to create a limitType
    def make_limit(minimum, center, maximum):
        limit = ismrmrd.xsd.limitType()
        limit.minimum = minimum
        limit.center = center
        limit.maximum = maximum
        return limit

    # Create encoding limits
    limits = ismrmrd.xsd.encodingLimitsType()

    # k-space encoding steps
    limits.kspace_encoding_step_1 = make_limit(0, n_phase_encoding_1 // 2, n_phase_encoding_1 - 1)
    limits.kspace_encoding_step_2 = make_limit(0, n_phase_encoding_2 // 2, n_phase_encoding_2 - 1)

    # Other limits
    param_map = {
        'contrast': 'number_of_echoes',
        'average': 'number_of_signal_averages',
        'phase': 'number_of_cardiac_phases',
        'slice': 'number_of_locations',
        'repetition': 'number_of_dynamic_scans',
        'user_0': 'number_of_extra_attribute_1_values',
        'user_1': 'number_of_extra_attribute_2_values',
    }

    for attr, param_name in param_map.items():
        max_val = int(_extract_single_parameter(general_info, param_name))
        setattr(limits, attr, make_limit(0, 0, max_val))

    # Assign to encoding
    encoding.encodingLimits = limits
    header.encoding.append(encoding)

    dataset.write_xml_header(header.toXML('utf-8'))

    # Create an acquisition and reuse it
    acq = ismrmrd.Acquisition()
    acq.resize(n_freq_encoding, sys.receiverChannels, trajectory_dimensions=2)
    acq.version = 1
    acq.available_channels = sys.receiverChannels
    acq.center_sample = round(n_freq_encoding / 2)
    acq.read_dir = (1.0, 0.0, 0.0)
    acq.phase_dir = (0.0, 1.0, 0.0)
    acq.slice_dir = (0.0, 0.0, 1.0)
    warnings.warn('Orientation hardcoded, please change it if needed.', stacklevel=2)

    scan_counter = 0

    def reshape_kspace_data(samples, attributes):
        """Reshape k-space data based on the type and attributes."""
        if samples is None:
            raise ValueError(f'No samples found for type {type}')

        if attributes is None:
            raise ValueError(f'No attributes found for type {type}')

        # Calculate k-space ranges
        kspace_ranges = _calculate_kspace_dimension_ranges(attributes)
        n_k0 = len(kspace_ranges['kx'])
        n_coils = int(_extract_single_parameter(general_info, 'number of coil channels'))
        n_acq = len(attributes) // n_coils
        return np.reshape(samples, (n_acq, n_coils, n_k0))

    # Add noise data
    samples = reshape_kspace_data(samples_per_type['NOI'], attributes_per_type['NOI'])
    for idx_acq in range(samples.shape[0]):
        # Set some fields in the header
        acq.scan_counter = scan_counter

        # Clear all flags
        acq.clearAllFlags()
        acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

        # Set noise data
        acq.resize(samples.shape[-1], acq.available_channels, trajectory_dimensions=2 if n_phase_encoding_2 == 1 else 3)
        acq.data[:] = samples[idx_acq, :, :]

        dataset.append_acquisition(acq)
        scan_counter += 1

    # Add image data
    attributes = attributes_per_type['STD']
    samples = reshape_kspace_data(samples_per_type['STD'], attributes)

    for idx_acq in range(samples.shape[0]):
        # Set some fields in the header
        acq.scan_counter = scan_counter

        acq.idx.kspace_encode_step_1 = attributes.iloc[idx_acq * acq.available_channels]['ky'] + n_phase_encoding_1 // 2
        acq.idx.kspace_encode_step_2 = attributes.iloc[idx_acq * acq.available_channels]['kz'] + n_phase_encoding_2 // 2
        acq.idx.repetition = attributes.iloc[idx_acq * acq.available_channels]['dyn']
        acq.idx.phase = attributes.iloc[idx_acq * acq.available_channels]['card']
        acq.idx.contrast = attributes.iloc[idx_acq * acq.available_channels]['echo']
        acq.idx.slice = attributes.iloc[idx_acq * acq.available_channels]['loca']
        acq.idx.average = attributes.iloc[idx_acq * acq.available_channels]['aver']
        acq.idx.user_0 = attributes.iloc[idx_acq * acq.available_channels]['extr1']
        acq.idx.user_1 = attributes.iloc[idx_acq * acq.available_channels]['extr2']

        acq.clearAllFlags()
        if attributes.iloc[idx_acq * acq.available_channels]['sign'] == -1:
            acq.setFlag(ismrmrd.ACQ_IS_REVERSE)

        acq_data = (
            samples[idx_acq, :, :]
            * np.exp(1j * np.pi * (acq.idx.kspace_encode_step_1 - n_phase_encoding_1 // 2)).astype(dtype=np.complex64)
            * np.exp(1j * np.pi * (acq.idx.kspace_encode_step_2 - n_phase_encoding_2 // 2)).astype(dtype=np.complex64)
        )

        # Compensate for FOV/2 shift along phase encoding direction
        acq.resize(n_freq_encoding, acq.available_channels, trajectory_dimensions=2 if n_phase_encoding_2 == 1 else 3)
        acq.data[:] = acq_data
        acq.discard_pre = 0
        acq.discard_post = 0
        dataset.append_acquisition(acq)
        scan_counter += 1

    # Clean up
    dataset.close()
