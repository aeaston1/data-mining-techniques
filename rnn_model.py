import sys,getopt
import numpy as np
import pandas as pd
from utilities import classifier, load_dataframe_from_file, loss_acc_plots

import tensorflow
import keras
from sklearn.model_selection import train_test_split


def run_rnn(dataframe, user):
    '''
    Run RNN model using dataframe input.

    Parameters
    ----------
    dataframe : pandas dataframe
        input dataframe used for the training the model
    user : string
        string of the user to train the rnn model for

    Returns
    -------
    history : keras history object
        a record of training loss and metrics and validation loss and metrics

    Raises
    ------
    None
    '''

    # Get single user, this is for multiple users
    first_user_df = dataframe[dataframe.id == '{}'.format(user)].sort_index()
    unique_variables = first_user_df.variable.unique()
    aggregate_variables = np.delete(unique_variables,0)
    aggregate_variables = np.delete(aggregate_variables,0)

    # unique day dates
    unique_dates = \
            first_user_df.index.map(lambda x: x.strftime('%Y-%m-%d')).unique()
    unique_dates = pd.to_datetime(unique_dates)
    new_dataframe = pd.DataFrame(index=unique_dates, columns=unique_variables)

    #filling the new_dataframe with values
    #averaging
    for var in aggregate_variables:
        my_df = first_user_df.value[first_user_df.variable == var]
        day_grouper = my_df.groupby(pd.Grouper(freq='1D')).aggregate(np.mean)
        day_grouper = day_grouper.rename('{}'.format(var))
        for i,x in enumerate(day_grouper.index.values):
            new_dataframe.loc[x, var] = day_grouper[i]
    new_dataframe = new_dataframe.fillna(0)
    #call - boolean. if a call is placed in the day then 1, else 0
    #sms - boolean. if an sms is placed in the day then 1, else 0
    Y_array = new_dataframe['mood'].values
    X_array = new_dataframe.values
    # Y = np.reshape(Y_array, (Y_array.shape[0], Y_array.shape[1]))
    Y = np.reshape(Y_array, (Y_array.shape[0], 1))
    X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1],1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, \
                                                            random_state=0)

    model_, kfold = classifier(X_train,Y_train)
    history = \
       model_.fit(X_train, Y_train,
                  epochs=200,
                  batch_size=1, 
                  validation_data=(X_test,Y_test))

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

    dataframe = load_dataframe_from_file('rnn_dataframes/{}_preprocessed.pkl'\
                                        .format(csv_name))
    for user in dataframe.id.unique():
        history = run_rnn(dataframe, user)
        loss_acc_plots(history)
        print(user, len(dataframe[dataframe.id == user]))
        print(history.history['acc'][-1], history.history['val_acc'][-1])
        print(history.history['loss'][-1], history.history['val_loss'][-1])
        stop
if __name__ == "__main__":
    main(sys.argv[1:])
