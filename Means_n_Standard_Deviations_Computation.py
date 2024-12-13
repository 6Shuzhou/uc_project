import numpy as np
import os

def compute_means_n_stds(experiment):
    data_path = f"Experiments_Transformed_Selected_Subset\Experiment_{experiment}\\"
    training_data_path = f"Experiments_Transformed_Selected_Subset\Experiment_{experiment}\Training_Set\\"
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

compute_means_n_stds(2)
compute_means_n_stds(3)
