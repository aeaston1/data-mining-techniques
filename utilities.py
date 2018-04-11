import sys, getopt
import pandas as pd

def csv_to_dataframe(csv_file):
    '''
    Method to convert csv file in pandas dataframe

    Parameters
    ----------
    csv_file : csv file
        csv file to convert to pandas dataframe

    Returns
    -------
    dataframe : pandas dataframe
        pandas dataframe of input csv data

    Raises
    ------
    None
    '''
    dataframe = pd.read_csv('{}.csv'.format(csv_file))
    return dataframe

def main(argv):
    pass

if __name__ == '__main__':
    main()
