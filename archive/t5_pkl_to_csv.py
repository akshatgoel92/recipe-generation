import numpy as np
import pandas as pd
import pickle
import os






def load_data(path):
    '''
    ---------
    Load data
    ---------
    '''
    with open(path, 'rb') as f:
        df = pickle.load(f)
    
    return(df)


def write_data(df, path):
    '''
    ----------
    Write data
    ----------
    '''
    df.to_csv(path)


def main(path, dest_path):
    '''
    ---------
    Execute functions
    ---------
    '''
    df = load_data(path)

    write_data(df, dest_path)

    return(df)


if __name__ == '__main__':
    
    type_path = 'test'
    
    path = os.path.join('/Users/akshatgoel/Desktop/dataframes', type_path + '.pkl')

    dest_path = os.path.join('/Users/akshatgoel/Desktop/dataframes', type_path + '.csv')
    
    df = main(path, dest_path)