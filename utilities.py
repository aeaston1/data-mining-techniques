import sys, getopt
import pandas as pd
import numpy as np
import pickle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
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

def classifier(X,Y):
    '''
    Nonsense to deal with a KerasClassifier and save it to file
    '''
    NOF_Xsamples, NOF_timesteps, NOF_input_dim = X.shape
    NOF_Ysamples, NOF_outputs = Y.shape
    def baseline_model():
        model = Sequential()
        model.add(LSTM(4, activation='sigmoid' \
                              , input_shape=(NOF_timesteps,NOF_input_dim)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())
        return model

    model = KerasClassifier(build_fn=baseline_model,
                                epochs=50,
                                batch_size=20,
                                verbose=1)
    seed = 7
    np.random.seed(seed)
    kfold = KFold(n_splits=2,
                  shuffle=True,
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
