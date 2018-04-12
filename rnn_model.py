import sys,getopt
import numpy as np
import pandas as pd
from utilities import classifier, load_dataframe_from_file

import tensorflow
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, RNN, LSTM

def run_rnn(dataframe):
    '''
    Run RNN model using dataframe input.

    Parameters
    ----------
    dataframe : pandas dataframe
        input dataframe used for the training the model

    Returns
    -------
    history : keras history object
        a record of training loss and metrics and validation loss and metrics

    Raises
    ------
    None
    '''

    # Get single user, this is for multiple users
    # for user in dataframe.id.unique():
    #     print(user, len(dataframe[dataframe.id == user]))


    list_of_onehots_x = [#'ChargePoint_skey',
                     'day_time_block']
    list_of_onehots_y = ['next_connection']
    Y_cols = [x for x in dataframe.columns.tolist() \
        for y in list_of_onehots_y if x.startswith(y)]
    X_cols = [x for x in dataframe.columns.tolist() \
        for y in list_of_onehots_x if x.startswith(y)]

    Y_array = dataframe[Y_cols].values
    X_array = dataframe[X_cols].values
    Y = np.reshape(Y_array, (Y_array.shape[0], Y_array.shape[1]))
    X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, \
                                                            random_state=0)

    model_, kfold = classifier(X_train,Y_train)
    history = \
        model_.fit(X_train, Y_train, epochs=15, validation_data=(X_test,Y_test))

    return history

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:c:", ["csv_name="])
    except getopt.GetoptError:
        print('rnn_model.py -c <csv_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('rnn_model.py -c <csv_name>')
            sys.exit()
        elif opt in ("-c"):
            csv_name = arg
    print('')

    dataframe = load_dataframe_to_file('rnn_dataframes/{}_preprocessed'\
                                        .format(csv_name), new_df)
    history = run_rnn(dataframe)

if __name__ == "__main__":
    main(sys.argv[1:])
