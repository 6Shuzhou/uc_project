import numpy as np
import os
import argparse

def compute_means_n_stds(transformation_setting, size, experiment, fold):
    data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/"
    training_data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/"
    all_means = []
    all_stds = []

    for file in os.listdir(training_data_path):
        if '_image' in file:
            current_image = np.load(training_data_path + file)
            current_means = np.mean(current_image, axis=(1, 2))
            current_stds = np.std(current_image, axis=(1, 2))

            all_means.append(current_means)
            all_stds.append(current_stds)

    with open(data_path + "Means.txt", "w") as file:
        for m in np.mean(np.array(all_means), axis=0):
            file.write(f"{m}\n")
    
    with open(data_path + "Standard_Deviations.txt", "w") as file:
        for std in np.std(np.array(all_stds), axis=0):
            file.write(f"{std}\n")
            

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
    
    compute_means_n_stds(args.setting, args.size, args.experiment, args.fold)
