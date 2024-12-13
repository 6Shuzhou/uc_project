import pandas as pd
import netCDF4
import xarray as xr
import numpy as np
from pathlib import Path
from utils.settings.config import LINEAR_ENCODER, CROP_ENCODING

crop_encoding = {v: k for k, v in CROP_ENCODING.items()}
crop_encoding[0] = 'Background/Other'

def retrieve_unique_classes(experiment):
    train_set = pd.read_csv(f"Experiments_Dataframes\Experiment_{experiment}\Training_Set_Experiment_{experiment}.csv")
    val_set = pd.read_csv(f"Experiments_Dataframes\Experiment_{experiment}\Validation_Set_Experiment_{experiment}.csv")
    test_set = pd.read_csv(f"Experiments_Dataframes\Experiment_{experiment}\Test_Set_Experiment_{experiment}.csv")
    classes = set()

    for patch_file in train_set['Patch']:
        patch = netCDF4.Dataset(Path(f"Experiments_Selected_Subset\Experiment_{experiment}\Training_Set" + "\\" + patch_file), 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch['labels'])).labels.to_numpy()
        classes = classes | set(np.unique(labels))

    for patch_file in val_set['Patch']:
        patch = netCDF4.Dataset(Path(f"Experiments_Selected_Subset\Experiment_{experiment}\Validation_Set" + "\\" + patch_file), 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch['labels'])).labels.to_numpy()
        classes = classes | set(np.unique(labels))
    
    for patch_file in test_set['Patch']:
        patch = netCDF4.Dataset(Path(f"Experiments_Selected_Subset\Experiment_{experiment}\Test_Set" + "\\" + patch_file), 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch['labels'])).labels.to_numpy()
        classes = classes | set(np.unique(labels))

    return sorted(list(classes))

def compute_class_weights(experiment):
    train_set = pd.read_csv(f"Experiments_Dataframes\Experiment_{experiment}\Training_Set_Experiment_{experiment}.csv")
    unique_classes = retrieve_unique_classes(experiment)
    class_counts = {c: 0 for c in unique_classes}

    for patch_file in train_set['Patch']:
        patch = netCDF4.Dataset(Path(f"Experiments_Selected_Subset\Experiment_{experiment}\Training_Set" + "\\" + patch_file), 'r')
        labels = xr.open_dataset(xr.backends.NetCDF4DataStore(patch['labels'])).labels.to_numpy()
        classes, counts = np.unique(labels, return_counts=True)

        for i in range(len(classes)):
            if (classes[i] in LINEAR_ENCODER.keys()) and (classes[i] != 0):
                class_counts[classes[i]] += counts[i]

    total_count = sum(list(class_counts.values()))
    class_weights = {}

    print(f"Experiment {experiment}, Total Count: {total_count}, Class Counts: {class_counts}")

    for k, v in class_counts.items():
        if v != 0:
            class_weights[k] = total_count / (len(LINEAR_ENCODER) * v)
        else:
            class_weights[k] = 0    

    with open(f"Experiments_Selected_Subset\Experiment_{experiment}\Class_Weights.txt", "w") as file:
        for cw in class_weights.values():
            file.write(f"{cw}\n")

    with open(f"Experiments_Selected_Subset\Experiment_{experiment}\Encoded_Classes.txt", "w") as file:
        for c in class_weights.keys():
            file.write(f"{c}\n")

    with open(f"Experiments_Selected_Subset\Experiment_{experiment}\Classes.txt", "w") as file:
        for c in class_weights.keys():
            file.write(f"{crop_encoding[c]}\n")

compute_class_weights(2)
compute_class_weights(3)
