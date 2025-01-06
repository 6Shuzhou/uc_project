import pandas as pd
import argparse


def required_tiles(df, size, experiment, fold, subset): # Determines the Data of which Tiles is required
    print(f"Necessary Tiles for {subset} Set of Fold {fold} of Experiment {experiment} with Size {size}%")
    
    for year, tile in sorted(list(set(zip(df['Year'], df['Tile']))), key=lambda x: (x[0], x[1])):
        print(f"- Year: {year}, Tile: {tile}")
    
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Required Tiles Analysis of Selected Dataset')

    parser.add_argument('--size', type=int, required=True,
                        help='The Dataset Percentage Size')
    parser.add_argument('--experiment', type=int, choices=[2,3], required=True,
                        help='Choose Experiment')
    parser.add_argument('--fold', type=int, required=True,
                        help='The K-Fold')
    args = parser.parse_args()
    
    for subset in ['Training', 'Validation', 'Test']:
        current_df = pd.read_csv(f"Dataframes/{args.size}%/Experiment_{args.experiment}/Fold_{args.fold}/{subset}_Set.csv") # Relative Path in Project Folder
        required_tiles(current_df, args.size, args.experiment, args.fold, subset)
