import shutil 
import os
import pandas as pd

def select_n_move_files(df, experiment, subset):
    destination_path = f"Experiments_Selected_Subset\Experiment_{experiment}\{subset}_Set\\"

    print(f"Missing Files of {subset} Set for Experiment {experiment}:")

    for i in range(len(df)):
        if df.loc[i]['Patch'] not in os.listdir(destination_path):
            source_path = f"Experiments Dataset\{df.loc[i]['Year']}\{df.loc[i]['Tile']}\\"

            if df.loc[i]['Patch'] in os.listdir(source_path):
                shutil.copyfile(source_path + df.loc[i]['Patch'], destination_path + df.loc[i]['Patch'])
            else: 
                print(f"- Year: {df.loc[i]['Year']}, Tile: {df.loc[i]['Tile']}, File: {df.loc[i]['Patch']}")
    
    print("\n")

train_exp_2 = pd.read_csv("Experiments_Dataframes\Experiment_2\Training_Set_Experiment_2.csv")
val_exp_2 = pd.read_csv("Experiments_Dataframes\Experiment_2\Validation_Set_Experiment_2.csv")
test_exp_2 = pd.read_csv("Experiments_Dataframes\Experiment_2\Test_Set_Experiment_2.csv")
train_exp_3 = pd.read_csv("Experiments_Dataframes\Experiment_3\Training_Set_Experiment_3.csv")
val_exp_3 = pd.read_csv("Experiments_Dataframes\Experiment_3\Validation_Set_Experiment_3.csv")
test_exp_3 = pd.read_csv("Experiments_Dataframes\Experiment_3\Test_Set_Experiment_3.csv")

select_n_move_files(train_exp_2, 2, "Training")
select_n_move_files(val_exp_2, 2, "Validation")
select_n_move_files(test_exp_2, 2, "Test")
select_n_move_files(train_exp_3, 3, "Training")
select_n_move_files(val_exp_3, 3, "Validation")
select_n_move_files(test_exp_3, 3, "Test")
