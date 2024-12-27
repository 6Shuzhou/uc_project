import shutil 
import os
import pandas as pd
import argparse

def select_n_move_files(df, size, experiment, fold, subset):
    core_dataset_path = r'D:/UC_Project_Sen4AgriNet_Dataset/Regular_Dataset'
    #destination_path = f"Experiments_Datasets\{size}%\Experiment_{experiment}\Fold_{fold}\{subset}_Set\\"
    destination_path = f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%/Experiment_{experiment}/Fold_{fold}/{subset}_Set/"

    if not os.path.exists(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%"):
        os.mkdir(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%")
    
    if not os.path.exists(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%\Experiment_{experiment}"):
        os.mkdir(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%/Experiment_{experiment}")
    
    if not os.path.exists(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%/Experiment_{experiment}/Fold_{fold}"):
        os.mkdir(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%/Experiment_{experiment}/Fold_{fold}")
    
    if not os.path.exists(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%/Experiment_{experiment}/Fold_{fold}/{subset}_Set"):
        os.mkdir(f"D:/UC_Project_Sen4AgriNet_Dataset/Selected_Dataset/{size}%/Experiment_{experiment}/Fold_{fold}/{subset}_Set")

    for i in range(len(df)):
        if df.loc[i]['Patch'] not in os.listdir(destination_path):
            source_path = core_dataset_path + '\\' + f"{df.loc[i]['Year']}\{df.loc[i]['Tile']}\\"

            if df.loc[i]['Patch'] in os.listdir(source_path):
                shutil.copyfile(source_path + df.loc[i]['Patch'], destination_path + df.loc[i]['Patch'])
            else: 
                print(f"- Year: {df.loc[i]['Year']}, Tile: {df.loc[i]['Tile']}, File: {df.loc[i]['Patch']}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select Files of Dataset')

    parser.add_argument('--size', type=int, required=True,
                        help='The Dataset Percentage Size')
    parser.add_argument('--experiment', type=int, choices=[2,3], required=True,
                        help='Choose Experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='The K-Fold')
    args = parser.parse_args()

    for subset in ["Training", "Validation", "Test"]:
        current_df = pd.read_csv(f"Experiments_Dataframes\{args.size}%\Experiment_{args.experiment}\Fold_{args.fold}\{subset}_Set.csv")
        select_n_move_files(current_df, args.size, args.experiment, args.fold, subset)
