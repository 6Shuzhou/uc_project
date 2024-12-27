import pandas as pd
import numpy as np
import os

from utils.tools import common_labels
from utils.settings.mappings.mappings_cat import SAMPLE_TILES as CAT_TILES
from utils.settings.mappings.mappings_fr import SAMPLE_TILES as FR_TILES
from utils.settings.config import LINEAR_ENCODER

from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from skmultilearn.model_selection.iterative_stratification import IterativeStratification 

full_patch_dataset = pd.read_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")

def filter_dataframe_n_retieve_unique_labels(tiles, 
                                             years, 
                                             common_labels=None):
    data = full_patch_dataset.copy() 
    data_to_keep = data[(data['Year'].isin(years)) & (data['Tile'].isin(tiles))].reset_index(drop=True)
    data_to_keep['Unique Labels'] = data_to_keep['Labels'].apply(lambda x: list(set(np.unique(x)) & set(LINEAR_ENCODER.keys()) & common_labels))
    data_to_keep = data_to_keep[~data_to_keep['Unique Labels'].isin([[0], []])].reset_index(drop=True)
    
    return data_to_keep.drop(labels=['Labels'], axis=1)

def train_val_test_split(experiment=2, 
                         downsampling_percentage=10, 
                         n_folds=3,
                         random_state=42):
    train_tiles, train_years, test_tiles, test_years, common_lbls = None, None, None, None, None

    if not os.path.exists(f"Experiments_Dataframes/{downsampling_percentage}%/"):
        os.mkdir(f"Experiments_Dataframes/{downsampling_percentage}%/")
    
    if not os.path.exists(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}"):
        os.mkdir(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}")

    for k in range(n_folds):
        if not os.path.exists(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}/Fold_{k+1}"):
            os.mkdir(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}/Fold_{k+1}")

    if experiment == 2:
        train_tiles, train_years = set(CAT_TILES), set(['2019', '2020'])
        test_tiles, test_years = set(FR_TILES), set(['2019'])
    else:
        train_tiles, train_years = set(FR_TILES), set(['2019'])
        test_tiles, test_years = set(CAT_TILES), set(['2020'])
        
    common_lbls = common_labels(train_tiles | test_tiles)
    train_r, val_r, test_r = [60, 20, 20]

    training_data = filter_dataframe_n_retieve_unique_labels(train_tiles, 
                                                             train_years, 
                                                             common_lbls).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_data = filter_dataframe_n_retieve_unique_labels(test_tiles, 
                                                         test_years, 
                                                         common_lbls).sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_training_patches = training_data.shape[0]
    n_test_patches = test_data.shape[0]

    training_data_train_r = (train_r / (train_r + val_r) * 100)
    training_data_val_r = (val_r / (train_r + val_r) * 100)
    training_data_test_r = (test_r / (train_r + val_r) * 100)
    new_training_data_train_r = ((n_training_patches * training_data_train_r) * downsampling_percentage) / (n_training_patches * 100)
    test_size = (training_data_test_r / 100) * (n_training_patches * (downsampling_percentage / 100))
    new_training_data_test_r = (test_size / n_test_patches) * 100

    training_idxs = []
    val_idxs = []
    test_idxs = []
    remainder_datas = []
    mlb_training = MultiLabelBinarizer()
    labels_onehot_training = pd.DataFrame(mlb_training.fit_transform(training_data['Unique Labels']), columns=mlb_training.classes_)
    training_stratifier = MultilabelStratifiedShuffleSplit(n_splits=n_folds, 
                                                           test_size= 1 - (new_training_data_train_r / 100),
                                                           random_state=random_state)

    if downsampling_percentage != 100:
        for current_training_idx, current_remainder_idx in training_stratifier.split(training_data.values[:, np.newaxis], labels_onehot_training.values):
            current_remainder_data = training_data.iloc[current_remainder_idx, :]
            current_labels_onehot_remainder = labels_onehot_training.iloc[current_remainder_idx, :]
            current_new_training_data_val_r = ((n_training_patches * training_data_val_r) * downsampling_percentage) / (current_remainder_data.shape[0] * 100)

            current_val_stratifier = MultilabelStratifiedShuffleSplit(n_splits=n_folds, 
                                                                    test_size= 1 - (current_new_training_data_val_r / 100),
                                                                    random_state=random_state)
            current_val_idx, _ = next(current_val_stratifier.split(current_remainder_data.values[:, np.newaxis], 
                                                                current_labels_onehot_remainder))
            
            training_idxs.append(current_training_idx)
            val_idxs.append(current_val_idx)
            remainder_datas.append(current_remainder_data)
    else:
        for current_training_idx, current_val_idx in training_stratifier.split(training_data.values[:, np.newaxis], labels_onehot_training.values):
            training_idxs.append(current_training_idx)
            val_idxs.append(current_val_idx)
    
    mlb_test = MultiLabelBinarizer()
    labels_onehot_test = pd.DataFrame(mlb_test.fit_transform(test_data['Unique Labels']), columns=mlb_test.classes_)
    test_stratifier = MultilabelStratifiedShuffleSplit(n_splits=n_folds,
                                                       test_size= 1 - (new_training_data_test_r / 100),
                                                       random_state=random_state)
    
    for test_idx, _ in test_stratifier.split(test_data.values[:, np.newaxis], labels_onehot_test.values):
        test_idxs.append(test_idx)

    for k in range(n_folds):
        X_train = training_data.iloc[training_idxs[k], :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)
        X_test = test_data.iloc[test_idxs[k], :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)

        if downsampling_percentage == 100:
            X_val = training_data.iloc[val_idxs[k], :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)
        else:
            X_val = remainder_datas[k].iloc[val_idxs[k], :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)

        X_train.to_csv(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}/Fold_{k+1}/Training_Set.csv", index=False)
        X_val.to_csv(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}/Fold_{k+1}/Validation_Set.csv", index=False)
        X_test.to_csv(f"Experiments_Dataframes/{downsampling_percentage}%/Experiment_{experiment}/Fold_{k+1}/Test_Set.csv", index=False)

for dp in [5, 10, 25, 50, 75, 100]:
    for experiment in [2,3]:
        train_val_test_split(experiment=experiment,
                             downsampling_percentage=dp)
