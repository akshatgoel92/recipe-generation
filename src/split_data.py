import pandas as pd
import numpy as np
import os


def load_data(path):
    '''
    Input: Path where data is stored
    Output: Loaded data
    '''
    df = pd.read_csv(path)
    
    return(df)
    


def split_data(df, split_percent=0.8):
    '''
    Input: Path where data is stored
    Output: Loaded data
    '''
    threshold = np.round(split_percent*len(df)).astype(int)
    
    return(df.iloc[:threshold, :], df.iloc[threshold:, :])

    
def filter_df(df, keep):
    '''
    Filter data-frame 
    columns
    '''
    return(df.loc[:, keep])



def write_data(df, dest_path):
    '''
    Write data to file
    '''
    df.to_csv(dest_path)


def main():
    '''
    Execute
    '''
    n_to_keep = 15000
    src = os.path.join("data", "RAW_recipes.csv")
    dest_train = os.path.join("data", "train.csv")
    dest_val = os.path.join("data", "val.csv")
    keep = ["name", "ingredients", "steps"]


    df = load_data(src)
    df = df.loc[:n_to_keep, :]
    df_filtered = filter_df(df, keep)
    train, val = split_data(df_filtered)

    for df_split, dest in zip([train, val], [dest_train, dest_val]):
        write_data(df_split, dest)

if __name__ == '__main__':

    main()