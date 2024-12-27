import pandas as pd

df = pd.read_pickle("Experiments_Dataframes\Patch_Data_Frame.pkl")

print(df.groupby(['Year', 'Country', 'Tile']).size().reset_index(name='Number of Patches'))
