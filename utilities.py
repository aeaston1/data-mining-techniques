import sys, getopt
import pandas as pd
import numpy as np
import pickle
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
import matplotlib.pyplot as plt

def write_dataframe_to_file(path_to_file, dataframe):
    '''
    Writes a chosen dataframe to a location of your choice, as a pickle.

    Parameters
    ----------
    path_to_file : string
        path to desired pickle location
    dataframe : dataframe, object
        dataframe to pickle in the chosen location

    Returns
    -------
    None

    Raises
    ------
    None
    '''
    print('Writing dataframe to {}'.format(path_to_file))
    with open(path_to_file, 'wb') as pkl:
        pickle.dump(dataframe, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print('Dataframe written.')

def load_dataframe_from_file(path_to_file):
    '''
    Reads a chosen dataframe from a location of your choice.

    Parameters
    ----------
    path_to_file : string
        path from pickle location

    Returns
    -------
    dataframe
        chosen dataframe loaded from pickle file

    Raises
    ------
    None
    '''
    print('Loading dataframe {}'.format(path_to_file))
    with open(path_to_file, 'rb') as pkl:
        dataframe = pickle.load(pkl)
    print('Dataframe loaded.')
    return dataframe

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

def classifier(X,Y, optimizer):
    '''
    Nonsense to deal with a KerasClassifier and save it to file
    '''
    NOF_Xsamples, NOF_timesteps, NOF_input_dim = X.shape
    NOF_Ysamples, NOF_outputs = Y.shape
    def baseline_model(optimiser=optimizer):
        model = Sequential()
        model.add(LSTM(1, activation='softmax' \
                              , input_shape=(NOF_timesteps,NOF_input_dim)
                              ))
        model.add(Dense(NOF_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimiser,
                      metrics=['accuracy'])
        # print(model.summary())
        return model

    model = KerasClassifier(build_fn=baseline_model,
                                epochs=10, #used for running model with kfold
                                batch_size=1, #not with grid search
                                verbose=0)

    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=5,
                  shuffle=False,
                  random_state=seed)

    # batch_size = [10, 20, 40, 60, 80, 100]
    # epochs = [10, 50, 100]
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', \
    #                                                                     'Nadam']
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',\
    #                                 'glorot_uniform', 'he_normal', 'he_uniform']
    # activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', \
    #                                         'sigmoid', 'hard_sigmoid', 'linear']
    # weight_constraint = [1, 2, 3, 4, 5]
    # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # neurons = [1, 5, 10, 15, 20, 25, 30]
    # param_grid = dict(activation=activation)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid)

    return model, kfold

def regressor(X,Y):
    '''
    Nonsense to deal with a KerasRegressor and save it to file
    '''
    NOF_Xsamples, NOF_timesteps, NOF_input_dim = X.shape
    NOF_Ysamples, NOF_outputs = Y.shape
    def baseline_model():
        model = Sequential()
        model.add(LSTM(10, activation='sigmoid' \
                              , input_shape=(NOF_timesteps,NOF_input_dim)))
        model.add(Dense(1, activation='sigmoid')) # regressor
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model

    model = KerasRegressor(build_fn=baseline_model,
                                epochs=20,
                                batch_size=1,
                                verbose=0)
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=10,
                  shuffle=False,
                  random_state=seed)

    return model, kfold

def loss_acc_plots(history_):
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))
    ax0.plot(history_.history['loss'])
    ax0.plot(history_.history['val_loss'])
    ax0.set_title('model train vs validation loss')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')
    ax0.legend(['train', 'validation'], loc='upper right')

    ax1.plot(history_.history['acc'])
    ax1.plot(history_.history['val_acc'])
    ax1.set_title('model train vs validation acc')
    ax1.set_ylabel('acc')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper right')
    plt.show()

def main(argv):
    pass

if __name__ == '__main__':
    main()
