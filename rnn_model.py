import sys,getopt
import numpy as np
import pandas as pd
from utilities import classifier, regressor
from utilities import load_dataframe_from_file, loss_acc_plots
import datetime
import tensorflow
import keras
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def run_rnn_classifier(dataframe, user=None):
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

    #onehot encode mood
    onehots = new_dataframe['mood'].copy()
    df_with_dummies = pd.get_dummies(onehots,columns='mood')
    new_dataframe = new_dataframe.drop(labels='mood',axis=1)
    new_dataframe = pd.concat([new_dataframe,df_with_dummies], axis=1)

    #not all users with have the full range of mood values. added if not present
    mood_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    mood_cols = [x for x in new_dataframe.columns.values if x in mood_range]
    add_mood_cols = list(set(mood_range)-set(mood_cols))
    new_dataframe.columns = [str(x) for x in new_dataframe.columns]
    for x in add_mood_cols:
        new_dataframe['{}'.format(x)] = np.zeros(new_dataframe.shape[0])

    new_dataframe = new_dataframe.fillna(0)

    #scale call and sms sum in to range of 0 to 1
    scaler = MinMaxScaler()
    scaled_call = scaler.fit_transform(new_dataframe.call.values.reshape(-1,1))
    new_dataframe.call = scaled_call
    scaled_sms = scaler.fit_transform(new_dataframe.sms.values.reshape(-1,1))
    new_dataframe.sms = scaled_sms

    cols = sorted(set(new_dataframe.columns.values))[:10]
    Y_cols = sorted(set(cols), key=float)
    X_cols = [x for x in new_dataframe.columns.values if x not in Y_cols]
    Y_array = new_dataframe[Y_cols].values
    X_array = new_dataframe[X_cols].values
    Y = np.reshape(Y_array, (Y_array.shape[0], Y_array.shape[1]))
    # Y = np.reshape(Y_array, (Y_array.shape[0], 1))
    X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1],1))
    # X = np.reshape(X_array, (X_array.shape[0],1 ,1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, \
                                                            random_state=0)

    model_, kfold = classifier(X_train,Y_train)

    history = \
       model_.fit(X_train, Y_train,
                  epochs=10,
                  batch_size=3,
                  validation_data=(X_test,Y_test))

    #grid search results
    # grid_result = grid.fit(X, Y)

    return history, kfold

def run_rnn_regressor(dataframe, user=None):
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

    #onehot encode mood
    # onehots = new_dataframe['mood'].copy()
    # df_with_dummies = pd.get_dummies(onehots,columns='mood')
    # new_dataframe = new_dataframe.drop(labels='mood',axis=1)
    # new_dataframe = pd.concat([new_dataframe,df_with_dummies], axis=1)

    #not all users with have the full range of mood values. added if not present
    # mood_range = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    # mood_cols = [x for x in new_dataframe.columns.values if x in mood_range]
    # add_mood_cols = list(set(mood_range)-set(mood_cols))
    # new_dataframe.columns = [str(x) for x in new_dataframe.columns]
    # for x in add_mood_cols:
    #     new_dataframe['{}'.format(x)] = np.zeros(new_dataframe.shape[0])

    #fill nan moods with 1 value
    # new_dataframe = new_dataframe.mood.fillna(1)
    new_dataframe = new_dataframe.fillna(0)


    #scale call, sms and mood sum in to range of 0 to 1
    scaler = MinMaxScaler()
    scaled_call = scaler.fit_transform(new_dataframe.call.values.reshape(-1,1))
    new_dataframe.call = scaled_call
    scaled_sms = scaler.fit_transform(new_dataframe.sms.values.reshape(-1,1))
    new_dataframe.sms = scaled_sms
    new_dataframe.mood = new_dataframe.mood/10

    # cols = sorted(set(new_dataframe.columns.values))[:10]
    # Y_cols = sorted(set(cols), key=float)
    X_cols = [x for x in new_dataframe.columns.values if x != 'mood']
    Y_array = new_dataframe['mood'].values
    X_array = new_dataframe[X_cols].values
    # Y = np.reshape(Y_array, (Y_array.shape[0], Y_array.shape[1]))
    Y = np.reshape(Y_array, (Y_array.shape[0], 1))
    X = np.reshape(X_array, (X_array.shape[0], X_array.shape[1],1))
    # X = np.reshape(X_array, (X_array.shape[0],1 ,1))


    # STRATIFIED SAMPLING, CLASS IMBALANCE, SPLIT TRAIN AND TEST SET CLEVERLY
    # AND THEN CHECK THE DISTRIBUTION OF 'MOOD' AMONG THEM.
    # CLASS IMBALANCE OF MOOD SCORES --> STRATIFIED SAMPLING
    # FOR CLASS IMBALANCE MAKE HISTOGRAM OF THE MOOD SCORES AND THEIR FREQ
    # USE BOOTSTRAPPING HERE TO MAKE A GOOD ESTIMATION OF THE ERROR (BOOK P.156)
    # SIGNIFICANCE TEST TO COMPARE MODELS
    #
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, \
                                                            random_state=0)

    model_ = regressor(X_train,Y_train)

    history = \
       model_.fit(X_train, Y_train,
                  epochs=100,
                  batch_size=1,
                  validation_data=(X_test,Y_test))

    #grid search results
    # grid_result = grid.fit(X, Y)

    return history, model_, X_test, Y_test, kfold

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
    history, kfold = \
                                    run_rnn_classifier(dataframe)
    # results = cross_val_score(model_, X_test, Y_test, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % \
    #                                 (results.mean()*100, results.std()*100))
    if history:
        loss_acc_plots(history)
        print("Accuracy : Validation Accuracy")
        print(history.history['acc'][-1], history.history['val_acc'][-1])
        print("Loss : Validation Loss")
        print(history.history['loss'][-1], history.history['val_loss'][-1])
    if grid_result:
        print("Best: %f using %s" % \
                                 (grid_result.best_score_,grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    main(sys.argv[1:])
