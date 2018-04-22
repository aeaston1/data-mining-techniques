import sys, getopt
import numpy as np
import pandas as pd
from utilities import load_dataframe_from_file

def run_benchmark_model(dataframe):
    '''The benchmark model implemented here predicts that the mood of the next
    day is the same as the mood of the day before.

    Parameters
    ----------
    dataframe : pandas dataframe
        input dataframe

    Returns
    -------
    accuracy : float
        accuracy of the predicted benchmark model

    Raises
    ------
    None
    '''
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

    mood_sf = new_dataframe.mood.apply(np.round)
    mood_df = pd.DataFrame({'time':mood_sf.index, 'mood':mood_sf.values})
    mood_df = mood_df.set_index('time')
    next_day_mood = mood_df.values[1:]
    next_day_mood = np.append(next_day_mood, 1)
    mood_df['next_day_mood'] = next_day_mood
    mood_df['true'] = (mood_df.mood == mood_df.next_day_mood)
    true_vals = mood_df.true.values
    accuracy = (np.where(true_vals == True)[0].shape[0]) / len(true_vals)
    return accuracy

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:c:", ["csv_name="])
    except getopt.GetoptError:
        print('benchmark_model.py -c <csv_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('benchmark_model.py -c <csv_name>')
            sys.exit()
        elif opt in ("-c"):
            csv_name = arg
    print('')

    dataframe = load_dataframe_from_file('rnn_dataframes/{}_preprocessed.pkl'\
                                        .format(csv_name))

    accuracy = run_benchmark_model(dataframe)
    print(accuracy)


if __name__ == "__main__":
    main(sys.argv[1:])
