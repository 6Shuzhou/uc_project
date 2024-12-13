import pandas as pd
import numpy as np

from utils.tools import common_labels
from utils.settings.mappings.mappings_cat import SAMPLE_TILES as CAT_TILES
from utils.settings.mappings.mappings_fr import SAMPLE_TILES as FR_TILES
from utils.settings.config import LINEAR_ENCODER

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection.iterative_stratification import IterativeStratification

full_patch_dataset = pd.read_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")

def filter_dataframe_n_retieve_unique_labels(tiles, years, common_labels=None):
    data = full_patch_dataset.copy() 
    data_to_keep = data[(data['Year'].isin(years)) & (data['Tile'].isin(tiles))].reset_index(drop=True)
    data_to_keep['Unique Labels'] = data_to_keep['Labels'].apply(lambda x: list(set(np.unique(x)) & set(LINEAR_ENCODER.keys()) & common_labels))
    data_to_keep = data_to_keep[~data_to_keep['Unique Labels'].isin([[0], []])].reset_index(drop=True)
    
    return data_to_keep.drop(labels=['Labels'], axis=1)

def train_val_test_split(experiment=2, downsampling_rate=0.1):
    train_tiles, train_years, test_tiles, test_years, common_lbls = None, None, None, None, None

    if experiment == 2:
        train_tiles, train_years = set(CAT_TILES), set(['2019', '2020'])
        test_tiles, test_years = set(FR_TILES), set(['2019'])
    else:
        train_tiles, train_years = set(FR_TILES), set(['2019'])
        test_tiles, test_years = set(CAT_TILES), set(['2020'])
        
    common_lbls = common_labels(train_tiles | test_tiles)
    train_r, val_r, test_r = [60, 20, 20]

    training_data = filter_dataframe_n_retieve_unique_labels(train_tiles, train_years, common_lbls)
    test_data = filter_dataframe_n_retieve_unique_labels(test_tiles, test_years, common_lbls)
    n_training_patches = training_data.shape[0]
    n_test_patches = test_data.shape[0]

    mlb_training = MultiLabelBinarizer()
    labels_onehot_training = pd.DataFrame(mlb_training.fit_transform(training_data['Unique Labels']), columns=mlb_training.classes_)

    new_train_r = ((n_training_patches * downsampling_rate) * train_r) / n_training_patches 

    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_train_r / 100), new_train_r / 100])
    train_idx, remainder_idx = next(stratifier.split(training_data.values[:, np.newaxis], labels_onehot_training.values))

    remainder_data = training_data.iloc[remainder_idx, :]
    labels_onehot_remainder = labels_onehot_training.iloc[remainder_idx, :]
    new_val_r = ((n_training_patches * downsampling_rate) * val_r) / remainder_data.shape[0]

    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_val_r / 100), new_val_r / 100])
    val_idx, _ = next(stratifier.split(remainder_data.values[:, np.newaxis], labels_onehot_remainder))

    test_size = (test_r / 100) * (n_training_patches * downsampling_rate)
    new_test_r = (test_size / n_test_patches) * 100

    mlb_test = MultiLabelBinarizer()
    labels_onehot_test = pd.DataFrame(mlb_test.fit_transform(test_data['Unique Labels']), columns=mlb_test.classes_)

    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - (new_test_r / 100), new_test_r / 100])
    test_idx, _ = next(stratifier.split(test_data.values[:, np.newaxis], labels_onehot_test.values))

    X_train = training_data.iloc[train_idx, :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)
    X_val = remainder_data.iloc[val_idx, :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)
    X_test = test_data.iloc[test_idx, :].sort_values(by=['Year', 'Tile']).reset_index(drop=True)

    X_train.to_csv(f"Experiments_Dataframes\Experiment_{experiment}\Training_Set_Experiment_{experiment}.csv", index=False)
    X_val.to_csv(f"Experiments_Dataframes\Experiment_{experiment}\Validation_Set_Experiment_{experiment}.csv", index=False)
    X_test.to_csv(f"Experiments_Dataframes\Experiment_{experiment}\Test_Set_Experiment_{experiment}.csv", index=False)

train_val_test_split(downsampling_rate=0.05)
train_val_test_split(experiment=3, downsampling_rate=0.05)
