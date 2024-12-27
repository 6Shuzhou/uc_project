import os
import pandas as pd
import netCDF4
import xarray as xr
from pathlib import Path

country_dict = {'31TBF':'Catalonia', 
                '31TCF':'Catalonia', 
                '31TCG':'Catalonia', 
                '31TDF':'Catalonia', 
                '31TDG':'Catalonia',
                '31TCJ':'France', 
                '31TDK':'France', 
                '31TCL':'France', 
                '31TDM':'France', 
                '31UCP':'France', 
                '31UDR':'France'}

dataset_path = r'D:\UC_Project_Sen4AgriNet_Dataset\Regular_Dataset' # Specify Path to Dataset

if not os.path.exists("Experiments_Dataframes\Patch_Data_Frame.pkl"): # Create and Save Dataframe to File in case it doesn't exist
    pd.DataFrame(columns=['Year',
                          'Country', 
                          'Tile', 
                          'Patch',
                          'Labels']).to_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl") 

patch_df = pd.read_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl") # Read in current Dataframe
current_length = len(patch_df)

for year in os.listdir(dataset_path): # Save File Info of each File in Dataset
    for tile in os.listdir(dataset_path + '\\' + year):
        if len(patch_df[(patch_df['Year'] == year) & (patch_df['Tile'] == tile)]) == 0:
            for patch in os.listdir(dataset_path + '\\' + year + '\\' + tile):
                current_patch_data = netCDF4.Dataset(Path(dataset_path + '\\' + year + '\\' + tile + '\\' + patch), 'r')
                patch_labels = xr.open_dataset(xr.backends.NetCDF4DataStore(current_patch_data['labels']))
                patch_df.loc[len(patch_df)] = [year, country_dict[tile], tile, patch, patch_labels.labels.to_numpy()]
            
            print(f"Data in Year {year} of Tile {tile} Extracted")

print(patch_df.sort_values(by=['Year', 'Tile']).reset_index(drop=True))
print(sorted(list(set(zip(patch_df['Year'], patch_df['Tile']))), key=lambda x: (x[0], x[1])))

if current_length != len(patch_df):
    patch_df.to_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")
