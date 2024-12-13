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

if not os.path.exists("Experiments_Dataframes\Patch_Data_Frame.pkl"):
    pd.DataFrame(columns=['Year',
                          'Country', 
                          'Tile', 
                          'Patch',
                          'Labels']).to_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")

patch_df = pd.read_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")
current_length = len(patch_df)

for year in os.listdir('Experiments_Dataset'):
    for tile in os.listdir('Experiments_Dataset\\' + year):
        if len(patch_df[(patch_df['Year'] == year) & (patch_df['Tile'] == tile)]) == 0:
            for patch in os.listdir('Experiments_Dataset\\' + year + '\\' + tile):
                current_patch_data = netCDF4.Dataset(Path('Experiments_Dataset\\' + year + '\\' + tile + '\\' + patch), 'r')
                patch_labels = xr.open_dataset(xr.backends.NetCDF4DataStore(current_patch_data['labels']))
                patch_df.loc[len(patch_df)] = [year, country_dict[tile], tile, patch, patch_labels.labels.to_numpy()]

print(patch_df.sort_values(by=['Year', 'Tile']).reset_index(drop=True))
print(sorted(list(set(zip(patch_df['Year'], patch_df['Tile']))), key=lambda x: (x[0], x[1])))

if current_length != len(patch_df):
    patch_df.to_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")
