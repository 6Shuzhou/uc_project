import numpy as np
import os
import argparse


def compute_means_n_stds(transformation_setting, size, experiment, fold): # Retrieve Means and Standard Deviations for all Files per Channel
    data_info_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/" # Relative Path to Data Info Path
    training_data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/" # Absolute Path to Training Set of Transformed Selected Dataset, Relative Path in Project Folder to Training Set of Transformed Selected Dataset: Transformed_Selected_Sen4AgriNet_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/
    all_means = []
    all_stds = []

    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/"): # Create Path to Info of Transformed Selected Dataset
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/")
    
    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/"): 
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/")

    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/"): 
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/")
    
    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/"): 
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/")

    for file in os.listdir(training_data_path): # Retrieve Means and Standard Deviations of each Channel per File
        if '_image' in file:
            current_image = np.load(training_data_path + file)
            current_means = np.mean(current_image, axis=(1, 2))
            current_stds = np.std(current_image, axis=(1, 2))

            all_means.append(current_means)
            all_stds.append(current_stds)

    with open(data_info_path + "Means.txt", "w") as file: # Save Means of each Channel to File
        for m in np.mean(np.array(all_means), axis=0):
            file.write(f"{m}\n")
    
    with open(data_info_path + "Standard_Deviations.txt", "w") as file: # Save Standard Deviations of each Channel to File
        for std in np.std(np.array(all_stds), axis=0):
            file.write(f"{std}\n")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve Means and Standard Deviations of Transformed Selected Dataset')

    parser.add_argument('--size', type=int, required=True,
                        help='The Dataset Percentage Size')
    parser.add_argument('--experiment', type=int, choices=[2,3], required=True,
                        help='Choose Experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='The K-Fold')
    parser.add_argument('--setting', type=int, choices=[1,2,3], default=1, required=True,
                        help='Transformation Setting')
    args = parser.parse_args()
    
    compute_means_n_stds(args.setting, args.size, args.experiment, args.fold)
