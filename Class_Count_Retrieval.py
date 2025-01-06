import numpy as np
import argparse
import os
from utils.settings.config import LINEAR_ENCODER, CROP_ENCODING


def class_count_retrieval(transformation_setting, size, experiment, fold, subset):
    data_info_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/" # Relative Path to Data Info Path
    data_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Transformed_Selected_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/{subset}_Set/" # Absolute Path to Training Set of Transformed Selected Dataset, Relative Path in Project Folder to Training Set of Transformed Selected Dataset: Transformed_Selected_Sen4AgriNet_Dataset_{transformation_setting}/{size}%/Experiment_{experiment}/Fold_{fold}/Training_Set/
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

    for file in os.listdir(data_path): # Count labels 
        if '_labels' in file:
            current_labels = np.load(data_path + file)
        
            classes, counts = np.unique(current_labels, return_counts=True)

            for i in range(len(classes)):
                if classes[i] != 0:
                    class_counts[classes[i]] += counts[i]

    print(f"Size {size} Experiment {experiment} Fold {fold} Setting {transformation_setting} {subset} Set, Class Counts: {class_counts}")
    
    with open(data_info_path + f"Class_Counts_{subset}_Set.txt", "w") as file: # Save Counts to File
        for cc in class_counts.values():
            file.write(f"{cc}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieve Class Counts of Transformed Selected Dataset')

    parser.add_argument('--size', type=int, required=True,
                        help='The Dataset Percentage Size')
    parser.add_argument('--experiment', type=int, choices=[2,3], required=True,
                        help='Choose Experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='The K-Fold')
    parser.add_argument('--setting', type=int, choices=[1,2,3], default=1, required=True,
                        help='Transformation Setting')
    args = parser.parse_args()
    
    for subset in ["Training", "Validation", "Test"]: # Retrieve Counts per Subset
        class_count_retrieval(args.setting, args.size, args.experiment, args.fold, subset)
