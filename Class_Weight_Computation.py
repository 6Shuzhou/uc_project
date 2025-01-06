import numpy as np
import argparse
import os
from utils.settings.config import LINEAR_ENCODER, CROP_ENCODING


def compute_class_weights(transformation_setting, size, experiment, fold):
    data_info_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/" # Relative Path to Data Info Path
    training_data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/" # Absolute Path to Training Set of Transformed Selected Dataset, Relative Path in Project Folder to Training Set of Transformed Selected Dataset: Transformed_Selected_Sen4AgriNet_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/
    crop_encoding = {v: k for k, v in CROP_ENCODING.items()}
    crop_encoding[0] = 'Background/Other'

    selected_classes = [0] + list(LINEAR_ENCODER.keys())[:-1]
    class_names = {i: crop_encoding[c] for i,c in enumerate(selected_classes)}
    class_counts = {c: 0 for c in list(class_names.keys())[1:]}

    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/"): # Create Path to Info of Transformed Selected Dataset
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/")
    
    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/"): 
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/")

    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/"): 
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/")
    
    if not os.path.exists(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/"): 
        os.mkdir(f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/")

    for file in os.listdir(training_data_path): # Count labels 
        if '_labels' in file:
            current_labels = np.load(training_data_path + file)
        
            classes, counts = np.unique(current_labels, return_counts=True)

            for i in range(len(classes)):
                if classes[i] != 0:
                    class_counts[classes[i]] += counts[i]

    total_count = sum(list(class_counts.values()))
    class_weights = {}

    #print(f"Size {size} Experiment {experiment} Fold {fold} Setting {transformation_setting}, Total Count: {total_count}, Class Counts: {class_counts}")

    for k, v in class_counts.items(): # Determine Class Weights with Formula based on: https://github.com/Orion-AI-Lab/S4A-Models/blob/master/compute_class_weights.py (Sen4Agrinet Experiments)
        class_weights[k] = total_count / (len(class_counts) * v)

    print(f"Size {size} Experiment {experiment} Fold {fold} Setting {transformation_setting}, Class Weights: {class_weights}")
    
    # with open(data_info_path + "Class_Counts.txt", "w") as file:
    #     for cc in class_counts.values():
    #         file.write(f"{cc}\n")

    with open(data_info_path + "Class_Weights.txt", "w") as file: # Save Class Weights to File
        for cw in class_weights.values():
            file.write(f"{cw}\n")

    with open(data_info_path + "Classes.txt", "w") as file: # Save Classes to File
        for c in class_weights.keys():
            file.write(f"{class_names[c]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve Classes and Class Weights of Transformed Selected Dataset')

    parser.add_argument('--size', type=int, required=True,
                        help='The Dataset Percentage Size')
    parser.add_argument('--experiment', type=int, choices=[2,3], required=True,
                        help='Choose Experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='The K-Fold')
    parser.add_argument('--setting', type=int, choices=[1,2,3], default=1, required=True,
                        help='Transformation Setting')
    args = parser.parse_args()
    
    compute_class_weights(args.setting, args.size, args.experiment, args.fold)
