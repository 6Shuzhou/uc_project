import pandas as pd

def required_tiles(df, experiment, subset):
    print(f"Necessary Tiles for {subset} Set of Experiment {experiment}")
    
    for year, tile in sorted(list(set(zip(df['Year'], df['Tile']))), key=lambda x: (x[0], x[1])):
        print(f"- Year: {year}, Tile: {tile}")
    
    print('\n')

train_exp_2 = pd.read_csv("Experiments_Dataframes\Experiment_2\Training_Set_Experiment_2.csv")
val_exp_2 = pd.read_csv("Experiments_Dataframes\Experiment_2\Validation_Set_Experiment_2.csv")
test_exp_2 = pd.read_csv("Experiments_Dataframes\Experiment_2\Test_Set_Experiment_2.csv")

train_exp_3 = pd.read_csv("Experiments_Dataframes\Experiment_3\Training_Set_Experiment_3.csv")
val_exp_3 = pd.read_csv("Experiments_Dataframes\Experiment_3\Validation_Set_Experiment_3.csv")
test_exp_3 = pd.read_csv("Experiments_Dataframes\Experiment_3\Test_Set_Experiment_3.csv")

required_tiles(train_exp_2, 2, "Training")
required_tiles(train_exp_2, 2, "Validation")
required_tiles(train_exp_2, 2, "Test")
required_tiles(train_exp_3, 3, "Training")
required_tiles(train_exp_3, 3, "Validation")
required_tiles(train_exp_3, 3, "Test")
