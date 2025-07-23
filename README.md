# Convert Philips raw data to (ISMR)MRD file format

This package extends the Julia package PhilipsDataList (https://github.com/oscarvanderheide/PhilipsDataList.jl) to enable the export of Philips raw MR data in *.{data,list}-format to ISMRMRD (https://github.com/ismrmrd/ismrmrd). 

## Use

If you have the files `raw.data` and `raw.list` in `/your/raw/data/folder`, then you can simply call

```python
philips_to_ismrmrd('/your/raw/data/folder/raw')
```

The converted (ISMR)MRD file will then we saved in `/your/raw/data/folder/raw.mrd`.

## Limitations

A valid (ISMR)MRD is created but the conversion to is optimized for a reconstruction of the raw data using [MRpro](https://github.com/PTB-MR/mrpro/tree/main). Not all required parameters are available in the *.{data,list} data and are hardcoded using reasonable values. The most important include the encoded and reconstructed field-of-view and the orientation. Also sequence parameters such as echo time, repetition time or flip angles are not available. 

## Development

- Create new environment, e.g. `conda create -n philips_ismrmrd python=3.12` 
- Install project by running `pip install -e .`

