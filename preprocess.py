import sys, getopt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utilities import csv_to_dataframe, write_dataframe_to_file

def preprocess(dataframe):
    '''
    Method to preprocess the advanced Data Mining
    Techniques dataset

    Parameters
    ----------
    dataframe : pandas dataframe
        input dataframe to be preprocessed

    Returns
    -------
    dataframe : pandas dataframe
        preprocessed dataframe

    Raises
    ------
    None
    '''
    #drop this useless column
    dataframe = dataframe.drop('Unnamed: 0', axis=1)
    #put the time column to pandas Datetime
    dataframe.time = pd.to_datetime(dataframe.time)
    #Search for missing and null values
    dataframe = dataframe.dropna(subset=['id', 'time', 'variable', 'value'], \
                                 how='any')
    #scale the inputs in range 0->1 for each variable
    scaler = MinMaxScaler()
    for x in dataframe.variable.unique():
        if not x in ['call', 'sms', 'activity', 'mood']:
            dataframe.value[dataframe.variable == x] = \
                scaler.fit_transform(dataframe.value[dataframe.variable == x] \
                .values.reshape(-1,1))
        else:
            continue
    dataframe = dataframe.set_index(dataframe.time)
    dataframe = dataframe.drop('time', axis=1)
    return dataframe

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:c:", ["csv_name="])
    except getopt.GetoptError:
        print('preprocess.py -c <csv_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('preprocess.py -c <csv_name>')
            sys.exit()
        elif opt in ("-c"):
            csv_name = arg
    print('')

    dataframe = csv_to_dataframe(csv_name)
    new_df = preprocess(dataframe)
    write_dataframe_to_file('rnn_dataframes/{}_preprocessed.pkl'\
                            .format(csv_name), new_df)
if __name__ == "__main__":
    main(sys.argv[1:])
