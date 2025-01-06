import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as cm


df = pd.read_pickle("Dataframes/Patch_Data_Frame.pkl")

number_of_patches_per_year_country_tile_df = df.groupby(['Year', 'Country', 'Tile']).size().reset_index(name='Number of Patches') # Retrieve Number of Patches per Year/Country/Tile
number_of_patches_per_year_country_tile_df_latex = number_of_patches_per_year_country_tile_df.to_latex(index=False)

number_of_patches_per_year_country_tile_df.to_csv("Exploratory_Data_Analysis_Files/Number_of_Patches_per_Year_Country_Tile.csv")
with open("Exploratory_Data_Analysis_Files/Number_of_Patches_per_Year_Country_Tile_Latex.txt", "w") as file:
    file.write(number_of_patches_per_year_country_tile_df_latex)

sizes = []
experiments = []
subsets = []
counts = [[], [], [], [], [], [], [], [], [], [], []]

with open("Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_1/5%/Experiment_2/Fold_1/Classes.txt", 'r') as classes_file: 
    classes = classes_file.read().split("\n")[:-1]

for s in [5,10,25,50]: # Retrieve Class Counts per Data Subset
    for i, e in enumerate(["Spatial", "Spatio-Temporal"]):
        for sb in ["Training", "Validation", "Test"]:
            current_counts_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_1/{s}%/Experiment_{i+2}/Fold_1/Class_Counts_{sb}_Set.txt"

            with open(current_counts_path, 'r') as counts_file:
                counts_set = list(map(int, counts_file.read().split("\n")[:-1]))

            sizes.append(str(s) + '%')
            experiments.append(e)
            subsets.append(sb)
            
            for j in range(len(counts_set)):
                counts[j].append(counts_set[j])
            
            plt.figure(figsize=(13,10))
            plt.barh(classes, counts_set)
            plt.title(f"Count Distribution of {sb} Set of {e} Experiment Dataset with Size {s}%", fontsize=20, ha='center', pad=20)
            plt.xlabel('Count', fontsize=15)
            plt.savefig(f"Exploratory_Data_Analysis_Files/Class_Counts_Distribution_Plots/Class_Counts_Distribution_Size_{s}_{e}_Experiment_{sb}_Set.png")
            plt.show()

class_counts_per_size_experiment_subset_df = pd.DataFrame({"Size":sizes, "Experiment":experiments, "Subset":subsets} | {c:count for c,count in zip(classes,counts)})
class_counts_per_size_experiment_subset_df_latex = class_counts_per_size_experiment_subset_df.to_latex(index=False)

class_counts_per_size_experiment_subset_df.to_csv("Exploratory_Data_Analysis_Files/Class_Counts_per_Size_Experiment_Subset.csv")
with open("Exploratory_Data_Analysis_Files/Class_Counts_per_Size_Experiment_Subset_Latex.txt", "w") as file:
    file.write(class_counts_per_size_experiment_subset_df_latex)

example_spectral_imagery = np.load("Exploratory_Data_Analysis_Files/Example_Transformed_Patch/2019_31TCJ_patch_24_09_image.npy") 
example_labeling = np.load("Exploratory_Data_Analysis_Files/Example_Transformed_Patch/2019_31TCJ_patch_24_09_labels.npy")

for i, band in enumerate(['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']): # Visualize Example Transformed Patch
    plt.figure(figsize=(12, 10)) 
    plt.imshow(example_spectral_imagery[i+7, :, :], cmap='viridis')
    plt.colorbar()
    plt.title(f"Spectral Band {band} of a Single Patch at a Single Timestep", fontsize=20,  pad=20)
    plt.savefig(f"Exploratory_Data_Analysis_Files/Example_Transformed_Patch_Visualization/{band}.png")
    plt.show()

label_mapping = {0:"Background/Other",
                 1:"Wheat",
                 2:"Maize",
                 3:"Sorghum",
                 4:"Barley",
                 5:"Rye",
                 6:"Oats",
                 7:"Grapes",
                 8:"Rapeseed",
                 9:"Sunflower",
                 10:"Potatoes",
                 11:"Peas"}

fig = plt.figure(figsize=(12,10))
colormap = cm.get_cmap('tab20', len(label_mapping))
plt.imshow(example_labeling[0, :, :], cmap=colormap, interpolation='none', alpha=0.5)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap(k), markersize=10, label=f"{v}") for k,v in label_mapping.items()]
plt.legend(handles=handles, loc='upper right', title="Classes", frameon=False, title_fontproperties={'weight':'bold'})
plt.title("Pixel-Wise Labelling of a Single Patch", fontsize=20)
plt.savefig("Exploratory_Data_Analysis_Files/Example_Transformed_Patch_Visualization/Labels.png")
plt.show()
