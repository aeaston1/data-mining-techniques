import sys,getopt
import numpy as np
import pandas as pd
from utilities import load_dataframe_from_file
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run_nonrnn_classifier(dataframe, user=None):
    '''
    Run RNN model using dataframe input.

    Parameters
    ----------
    dataframe : pandas dataframe
        input dataframe used for the training the model
    user : string
        string of the user to train the rnn model for if specifying user

    Returns
    -------
    history : keras history object
        a record of training loss and metrics and validation loss and metrics

    Raises
    ------
    None
    '''

    # this is for multiple users
    # first_user_df = dataframe[dataframe.id == '{}'.format(user)].sort_index()
    #generalised for all users
    first_user_df = dataframe.sort_index().copy()

    unique_variables = first_user_df.variable.unique()
    aggregate_variables = np.delete(unique_variables,0)
    aggregate_variables = np.delete(aggregate_variables,0)

    # unique day dates SLOW
    unique_dates = \
            first_user_df.index.map(lambda x: x.strftime('%Y-%m-%d')).unique()
    unique_dates = pd.to_datetime(unique_dates)
    new_dataframe = pd.DataFrame(index=unique_dates,columns=aggregate_variables)

    #filling the new_dataframe with values
    #averaging SLOW
    for var in aggregate_variables:
        my_df = first_user_df.value[first_user_df.variable == var]
        day_grouper = my_df.groupby(pd.Grouper(freq='1D')).aggregate(np.mean)
        day_grouper = day_grouper.rename('{}'.format(var))
        for i,x in enumerate(day_grouper.index.values):
            new_dataframe.loc[x, var] = day_grouper[i]

    #summation SLOW
    for var in ['call', 'sms']:
        my_df = first_user_df.value[first_user_df.variable == var]
        day_grouper = my_df.groupby(pd.Grouper(freq='1D')).aggregate(np.sum)
        day_grouper = day_grouper.rename('{}'.format(var))
        for i,x in enumerate(day_grouper.index.values):
            new_dataframe.loc[x, var] = day_grouper[i]

    new_dataframe.mood = new_dataframe.mood.apply(np.round)
    new_dataframe = new_dataframe.fillna(0)
    #scale call and sms sum in to range of 0 to 1
    scaler = MinMaxScaler()
    scaled_call = scaler.fit_transform(new_dataframe.call.values.reshape(-1,1))
    new_dataframe.call = scaled_call
    scaled_sms = scaler.fit_transform(new_dataframe.sms.values.reshape(-1,1))
    new_dataframe.sms = scaled_sms
    X = new_dataframe.iloc[:, new_dataframe.columns != 'mood']
    Y = new_dataframe.mood

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, \
                                                            random_state=0)
    total_history = []
    for k_splits in range(1,21,1):
        clf = neighbors.KNeighborsClassifier(k_splits, weights='uniform')
        history = clf.fit(X_train, Y_train)
        total_history.append(history.score(X_test, Y_test))

    return total_history

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:c:", ["csv_name="])
    except getopt.GetoptError:
        print('nonrnn_model.py -c <csv_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('nonrnn_model.py -c <csv_name>')
            sys.exit()
        elif opt in ("-c"):
            csv_name = arg
    print('')

    dataframe = load_dataframe_from_file('rnn_dataframes/{}_preprocessed.pkl'\
                                        .format(csv_name))
    total_history = run_nonrnn_classifier(dataframe)
    fig, ax0 = plt.subplots(nrows=1)
    ax0.plot(total_history, 'o')
    ax0.set_title('K-Means clusters accuracy')
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('K-Clusters')
    ax0.xaxis.set_ticks(np.arange(0,21,1))
    ax0.set_xticklabels(np.arange(1,21,1))
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
