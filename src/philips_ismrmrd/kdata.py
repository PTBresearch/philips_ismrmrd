"""Sort and process k-space data."""

import re

import numpy as np
import xarray as xr  # For named dims;

from philips_ismrmrd.reader import read_data_list
from philips_ismrmrd.utils import DIMENSIONS_STD
from philips_ismrmrd.utils import _num_bytes_to_num_samples


def data_list_to_kspace(path_to_data_or_list, drop_dims=True, remove_readout_oversampling=False, offset_array=False):
    """Read raw data and attributes."""
    samples_per_type, attributes_per_type, general_info = read_data_list(path_to_data_or_list)

    samples = samples_per_type.get('STD')
    attributes = attributes_per_type.get('STD')

    kspace = samples_to_kspace(
        samples,
        attributes,
        general_info,
        drop_dims=drop_dims,
        remove_readout_oversampling=remove_readout_oversampling,
        offset_array=offset_array,
    )

    return kspace


def samples_to_kspace(
    samples, attributes, general_info, drop_dims=True, remove_readout_oversampling=False, offset_array=False
):
    """Convert samples and attributes to k-space."""
    if not all(attributes['typ'] == 'STD'):
        raise ValueError("All attributes must be type 'STD'.")

    if not _all_readouts_same_size(attributes):
        raise ValueError('All readouts must have the same size.')

    kspace = _samples_to_kspace(samples, attributes)

    if drop_dims:
        # Drop dims with size 1 in xarray by squeezing
        kspace = kspace.squeeze()

    if remove_readout_oversampling:
        oversampling_factor = _extract_readout_oversampling_factor(general_info)
        kspace = _remove_readout_oversampling(kspace, oversampling_factor)

    if not offset_array:
        kspace = _remove_custom_indices_keep_nameddims(kspace)

    return kspace


def _samples_to_kspace(samples, attributes):
    # Check again if needed
    if not all(attributes['typ'] == 'STD'):
        raise ValueError("Attributes must be type 'STD'.")

    if not _all_readouts_same_size(attributes):
        raise ValueError('Readouts must have the same size.')

    ranges = _calculate_kspace_dimension_ranges(attributes)
    kspace = _allocate_kspace(ranges)

    _fill_kspace(kspace, samples, attributes)

    return kspace


def _all_readouts_same_size(attributes):
    return attributes['size'].nunique() == 1


def _calculate_kspace_dimension_ranges(attributes):
    # Calculate number of samples per readout from size in bytes
    size_readout = attributes.iloc[0]['size']
    samples_per_readout = _num_bytes_to_num_samples(size_readout)

    range_kx = np.arange(-samples_per_readout // 2, samples_per_readout // 2)

    ranges = {}
    for dim in DIMENSIONS_STD:
        if dim == 'kx':
            ranges['kx'] = range_kx
        else:
            min_val = attributes[dim].min()
            max_val = attributes[dim].max()
            ranges[dim] = np.arange(min_val, max_val + 1)

    return ranges


def _allocate_kspace(ranges):
    # Calculate shape
    shape = tuple(len(ranges[dim]) for dim in DIMENSIONS_STD)

    # Create a DataArray with dims and coords
    coords = {dim: ranges[dim] for dim in DIMENSIONS_STD}
    kspace = xr.DataArray(np.zeros(shape, dtype=np.complex64), dims=DIMENSIONS_STD, coords=coords)

    return kspace


def _fill_kspace(kspace, samples, attributes):
    # Reshape samples into (kx, num_readouts)
    kx_len = kspace.sizes['kx']
    num_readouts = len(attributes)
    samples = np.reshape(samples, (kx_len, num_readouts))

    if attributes.shape[0] != samples.shape[1]:
        raise ValueError('Mismatch between number of readouts and samples.')

    print('Sorting data into k-space...')

    for i in range(num_readouts):
        # Extract indices for current readout (all dims except kx)
        idx = tuple(attributes.iloc[i][dim] for dim in DIMENSIONS_STD if dim != 'kx')

        # Insert samples[:, i] at the index in kspace
        # xarray supports .loc for indexing by coordinate values
        kspace.loc[dict(zip(DIMENSIONS_STD[1:], idx, strict=False))][:] = samples[:, i]

    # TODO: handle :sign multiplication if needed


def _extract_readout_oversampling_factor(general_info):
    for line in general_info:
        if 'kx_oversample_factor' in line:
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                val = float(match.group(1))
                if not np.isclose(val, round(val)):
                    raise ValueError(f'Oversampling factor {val} not an integer.')
                return int(round(val))
            else:
                raise ValueError(f'Could not parse oversampling factor from line: {line}')
    raise ValueError('kx_oversample_factor not found in general info.')


def _remove_readout_oversampling(kspace, oversampling_factor):
    n = kspace.sizes['kx']
    kx_max = n // (2 * oversampling_factor)

    # Remove offsets by converting coords to zero-based
    data = kspace.data

    # FFT shift and inverse FFT along kx axis (axis 0)
    tmp = np.fft.ifftshift(np.fft.ifft(data, axis=0), axes=0)

    # Crop central region in image space
    center_start = n // 2 - kx_max
    center_end = n // 2 + kx_max
    tmp_cropped = tmp[center_start:center_end, ...]

    # FFT back
    cropped_kspace = np.fft.fft(np.fft.fftshift(tmp_cropped, axes=0), axis=0)

    # Create new coordinate for kx
    new_kx = np.arange(-kx_max, kx_max)

    # Build new xarray DataArray with cropped data and updated coords
    coords = kspace.coords.to_index().to_dict()
    coords['kx'] = new_kx
    new_dims = kspace.dims

    new_kspace = xr.DataArray(cropped_kspace.astype(np.complex64), dims=new_dims, coords=coords)

    return new_kspace


def _remove_custom_indices_keep_nameddims(kspace):
    # Strip custom coords and reset to default zero-based indices but keep dims and shape
    shape = kspace.shape
    new_coords = {dim: np.arange(size) for dim, size in zip(kspace.dims, shape, strict=False)}
    new_kspace = xr.DataArray(kspace.data.copy(), dims=kspace.dims, coords=new_coords)
    return new_kspace
