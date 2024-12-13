import os
import netCDF4
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import xarray as xr

IMG_SIZE = 366
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

REFERENCE_BAND = 'B02'

def process_patch(source_path, file, out_path, num_buckets, bands, classes, padded_patch_height,
                  padded_patch_width, medians_dtype, label_dtype, group_freq, output_size,
                  pad_top, pad_bot, pad_left, pad_right):

    netcdf = netCDF4.Dataset((source_path + file), 'r')
    medians = get_medians(netcdf, 0, num_buckets, group_freq, bands, padded_patch_height,
                          padded_patch_width, output_size, pad_top, pad_bot,
                          pad_left, pad_right, medians_dtype)

    num_bins, num_bands = medians.shape[:2]
    medians = sliding_window_view(medians, [num_bins, num_bands, output_size[0], output_size[1]], [1, 1, output_size[0], output_size[1]]).squeeze()
    sub_idx = 1
    
    if output_size == [366, 366]:
        np.save(out_path + file[:-3]  + '_image', medians[3:9, :, :, :].reshape((6*medians.shape[1], medians.shape[2], medians.shape[3])).astype(medians_dtype))
    else:
        for i in range(medians.shape[0]):
            for j in range(medians.shape[1]):
                np.save(out_path + f"{file[:-3] + '_sub_patch_' + str(sub_idx) + '_image'}", medians[i, j, 3:9, :, :, :].reshape((6*medians.shape[3], medians.shape[4], medians.shape[5])).astype(medians_dtype))
                sub_idx+=1
            
    labels = get_labels(netcdf, classes, output_size, pad_top, pad_bot, pad_left, pad_right)
    labels = sliding_window_view(labels, output_size, output_size)
    labels = labels.squeeze() 

    lbl_idx = 1
    
    if output_size == [366, 366]:
        np.save(out_path + file[:-3] + '_labels', labels.reshape((1, labels.shape[0], labels.shape[1])).astype(label_dtype))
    else:
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                np.save(out_path + f"{file[:-3] + '_sub_patch_' + str(lbl_idx) + '_labels'}", labels[i, j, :, :].reshape((1, labels.shape[2], labels.shape[3])).astype(label_dtype))
                lbl_idx += 1

def sliding_window_view(arr, window_shape, steps):
    in_shape = np.array(arr.shape[-len(steps):])  
    window_shape = np.array(window_shape)  
    steps = np.array(steps)  
    nbytes = arr.strides[-1] 

    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    step_strides = tuple(window_strides[-len(steps):] * steps)
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)

    return as_strided(arr, shape=outshape, strides=strides, writeable=False)


def get_medians(netcdf, start_bin, window, group_freq, bands,
                padded_patch_height, padded_patch_width, output_size,
                pad_top, pad_bot, pad_left, pad_right, medians_dtype):
    year = netcdf.patch_year

    date_range = pd.date_range(start=f'{year}-01-01', end=f'{int(year) + 1}-01-01', freq=group_freq)

    medians = np.empty((len(bands), window, padded_patch_height, padded_patch_width), dtype=medians_dtype)

    for band_id, band in enumerate(bands):
        band_data = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf[band]))

        band_data = band_data.groupby_bins(
            'time',
            bins=date_range,
            right=True,
            include_lowest=False,
            labels=date_range[:-1]
        ).median(dim='time')

        band_data = band_data.resample(time_bins=group_freq).median(dim='time_bins')

        band_data = band_data.interpolate_na(dim='time_bins', method='linear', fill_value='extrapolate')

        band_data = band_data.isel(time_bins=slice(start_bin, start_bin + window))

        band_data = band_data[f'{band}'].values

        expand_ratio = int(BANDS[band] / BANDS[REFERENCE_BAND])

        if expand_ratio != 1:
            band_data = np.repeat(band_data, expand_ratio, axis=1)
            band_data = np.repeat(band_data, expand_ratio, axis=2)

        if  (output_size[0] < band_data.shape[1]) or (output_size[1] < band_data.shape[2]):
            band_data = np.pad(band_data,
                                pad_width=((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
                                mode='constant',
                                constant_values=0)

        medians[band_id, :, :, :] = np.expand_dims(band_data, axis=0)

    return medians.transpose(1, 0, 2, 3)   # (T, B, H, W)


def get_labels(netcdf, classes, output_size, pad_top, pad_bot, pad_left, pad_right):
    labels = xr.open_dataset(xr.backends.NetCDF4DataStore(netcdf['labels']))['labels'].values

    mapping = {c: i for i,c in enumerate(classes)}
    labels = np.vectorize(mapping.get)(labels)

    if (output_size[0] < labels.shape[0]) or (output_size[1] < labels.shape[1]):
        labels = np.pad(labels,
                        pad_width=((pad_top, pad_bot), (pad_left, pad_right)),
                        mode='constant',
                        constant_values=0)

    return labels


def get_padding_offset(patch_height, patch_width, output_size):
    img_size_x = patch_height
    img_size_y = patch_width

    output_size_x = output_size[0]
    output_size_y = output_size[1]

    if img_size_x >= output_size_x:
        pad_x = int(output_size_x - img_size_x % output_size_x)
    else:
        pad_x = output_size_x - img_size_x

    if img_size_y >= output_size_y:
        pad_y = int(output_size_y - img_size_y % output_size_y)
    else:
        pad_y = output_size_y - img_size_y

    if not pad_x == output_size_x:
        pad_top = int(pad_x // 2)
        pad_bot = int(pad_x // 2)

        if not pad_x % 2 == 0:
            pad_top += 1
    else:
        pad_top = 0
        pad_bot = 0

    if not pad_y == output_size_y:
        pad_left = int(pad_y // 2)
        pad_right = int(pad_y // 2)

        if not pad_y % 2 == 0:
            pad_left += 1
    else:
        pad_left = 0
        pad_right = 0

    return pad_top, pad_bot, pad_left, pad_right


def calculate_subpatches(output_size):
    assert output_size[0] == output_size[1], \
        f'Only square sub-patch size is supported. Mismatch: {output_size[0]} != {output_size[1]}.'

    patch_width, patch_height = IMG_SIZE, IMG_SIZE
    padded_patch_width, padded_patch_height = IMG_SIZE, IMG_SIZE

    if (output_size[0] == patch_height) or (output_size[1] == patch_width):
        return patch_height, patch_width, 0, 0, 0, 0

    if (patch_height % output_size[0] != 0) or (patch_width % output_size[1] != 0):
        requires_pad = True
        pad_top, pad_bot, pad_left, pad_right = get_padding_offset(patch_height, patch_width, output_size)

        padded_patch_height += (pad_top + pad_bot)
        padded_patch_width += (pad_left + pad_right)
    else:
        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0

    return padded_patch_height, padded_patch_width, pad_top, pad_bot, pad_left, pad_right

def convert_netcdf4_file(experiment, subset):
    core_path = f"Experiments_Selected_Subset\Experiment_{experiment}\\"
    source_path = f"Experiments_Selected_Subset\Experiment_{experiment}\{subset}_Set\\"
    destination_path = f"Experiments_Transformed_Selected_Subset\Experiment_{experiment}\{subset}_Set\\"
    encoded_classes_file = core_path + "Encoded_Classes.txt"
    medians_dtype = np.float32
    label_dtype = np.int16
    group_freq = '1MS'
    bands = ['B02', 'B03', 'B04', 'B08']
    output_size = [366, 366]
    #output_size = [61, 61]
    num_buckets = len(pd.date_range(start=f'2020-01-01', end=f'2021-01-01', freq=group_freq)) - 1
    padded_patch_height, padded_patch_width, pad_top, pad_bot, pad_left, pad_right = calculate_subpatches(output_size)

    with open(encoded_classes_file, "r") as file:
        file_content = file.read()

    classes = [int(c) for c in file_content.split('\n')[:-1]]
    all_net4cdf_files = os.listdir(source_path)

    print(f'Start Processing Files of Experiment {experiment} {subset} Set.')
    print(f'Saving Processed Files of Experiment {experiment} {subset} Set Into: {destination_path}.\n')

    for i, net4cdf_file in enumerate(all_net4cdf_files):
        print(f"- Processing File {i+1} of {len(all_net4cdf_files)} of Experiment {experiment} {subset} Set: {net4cdf_file}")
        
        process_patch(source_path=source_path,
                      file=net4cdf_file, 
                      out_path=destination_path, 
                      num_buckets=num_buckets, 
                      bands=bands, 
                      classes=classes,
                      padded_patch_height=padded_patch_height, 
                      padded_patch_width=padded_patch_width, 
                      pad_top=pad_top, 
                      pad_bot=pad_bot, 
                      pad_left=pad_left, 
                      pad_right=pad_right, 
                      medians_dtype=medians_dtype, 
                      label_dtype=label_dtype, 
                      group_freq=group_freq, 
                      output_size=output_size)

    print(f'\nFinished Processing Files of Experiment {experiment} {subset} Set.\n')

convert_netcdf4_file(2, "Training")
convert_netcdf4_file(2, "Validation")
convert_netcdf4_file(2, "Test")

convert_netcdf4_file(3, "Training")
convert_netcdf4_file(3, "Validation")
convert_netcdf4_file(3, "Test")
