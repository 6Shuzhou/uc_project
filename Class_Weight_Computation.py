import pandas as pd
import netCDF4
import xarray as xr
import numpy as np
import argparse
import os
from pathlib import Path
from utils.settings.config import LINEAR_ENCODER, CROP_ENCODING


def compute_class_weights(transformation_setting, size, experiment, fold):
    data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/"
    training_data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/"

    crop_encoding = {v: k for k, v in CROP_ENCODING.items()}
    crop_encoding[0] = 'Background/Other'

    selected_classes = [0] + list(LINEAR_ENCODER.keys())[:-1]
    class_names = {i: crop_encoding[c] for i,c in enumerate(selected_classes)}
    class_counts = {c: 0 for c in list(class_names.keys())[1:]}

    for file in os.listdir(training_data_path):
        if '_labels' in file:
            current_labels = np.load(training_data_path + file)
        
            classes, counts = np.unique(current_labels, return_counts=True)

            for i in range(len(classes)):
                if classes[i] != 0:
                    class_counts[classes[i]] += counts[i]

    total_count = sum(list(class_counts.values()))
    class_weights = {}

    print(f"Size {size} Experiment {experiment} Fold {fold} Setting {transformation_setting}, Total Count: {total_count}, Class Counts: {class_counts}")

    for k, v in class_counts.items():
        class_weights[k] = total_count / (len(class_counts) * v)

    print(class_weights)
    
    with open(data_path + "Class_Weights.txt", "w") as file:
        for cw in class_weights.values():
            file.write(f"{cw}\n")

    with open(data_path + "Classes.txt", "w") as file:
        for c in class_weights.keys():
            file.write(f"{class_names[c]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select Files of Dataset')

    parser.add_argument('--size', type=int, required=True,
                        help='The Dataset Percentage Size')
    parser.add_argument('--experiment', type=int, choices=[2,3], required=True,
                        help='Choose Experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='The K-Fold')
    parser.add_argument('--setting', type=int, choices=[1,2], default=1, required=True,
                        help='Transformation Setting')
    args = parser.parse_args()
    
    compute_class_weights(args.setting, args.size, args.experiment, args.fold)
