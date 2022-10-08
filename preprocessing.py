import pandas as pd


def train_test_split_by_days(filename: str,
                             train_days: int = 30,
                             test_days: int = 5,
                             window: int = 8,
                             time_step: tuple[int, str] = (60000, 'ms'),
                             sep: str = ';',
                             to_the_end: bool = False,
                             start: int = 0,
                             ):
    """
    :param filename:        Local path to csv/excel/etc. file that must be split
    :param train_days:      Desired number of days in train dataset
    :param test_days:       Desired number of days in test dataset
    :param window:          Prediction Window (PW) (to make sure that some lines overlap correctly)
    :param time_step:       (int(difference_in_time_between_neighbour_lines),
                                str('ms' (milliseconds) or 's' ('seconds')),)
    :param sep:             Pandas.read_csv(sep=)
    :param to_the_end:      Whether we take lines to the end of DataFrame or not
    :param start:           Starting index. Ignored if :to_the_end: is True
    :return:
    """
    step, mode = time_step
    steps_in_day = ((24 * 60 * 60) * (1000 if mode == 'ms' else 1)) // step

    data = pd.read_csv(filename, header=0, sep=sep)

    # check if the file is long enough
    assert len(data) >= steps_in_day * (train_days + test_days) + start + window

    if to_the_end:
        test_end_idx = len(data)  # first Y_test
        test_start_idx = test_end_idx - window - int(test_days * steps_in_day)  # last Y_test

        train_end_idx = test_start_idx + window  # last Y_train
        train_start_idx = train_end_idx - window - int(train_days * steps_in_day)  # first Y_train
    else:
        train_start_idx = start  # first Y_train
        train_end_idx = train_start_idx + int(train_days * steps_in_day) + window  # last Y_train

        test_start_idx = train_end_idx - window  # first Y_test
        test_end_idx = test_start_idx + int(test_days * steps_in_day) + window  # last Y_test

    train_data = data[train_start_idx:train_end_idx]
    test_data = data[test_start_idx:test_end_idx]

    # finding index of '.' for correct train\test filename assembling
    dot_idx = filename.index('.')
    train_data.to_csv(filename[:dot_idx] + '_train' + filename[dot_idx:], sep=';', index=False)
    test_data.to_csv(filename[:dot_idx] + '_test' + filename[dot_idx:], sep=';', index=False)

    train_filename = filename[:dot_idx] + "_train" + filename[dot_idx:]
    test_filename = filename[:dot_idx] + "_test" + filename[dot_idx:]
    print(f'Train data saved to file: {train_filename}, '
          f'shape of file: {train_data.shape}')
    print(f'Test data saved to file: {test_filename}, '
          f'shape of file: {test_data.shape}')

    return train_filename, test_filename


def column_multiplier(filename: str,
                      ignore_time: bool = True,
                      time_col: str = None,
                      backward: bool = False,
                      sep: str = ';',
                      dtype: str = 'float64',
                      root: str = 'data/column_multiplier/',
                      ):
    print('\n************************')
    print('STARTING DATA ' + ('BACKWARD MULTIPLICATION' if backward else 'DIVISION'))
    print('************************\n')

    # Can't ignore time if time_col is None
    assert not ignore_time or (time_col is not None and ignore_time)

    data = pd.read_csv(filename, header=0, sep=sep, dtype=dtype)
    dot_idx = filename.index('.')
    new_filename = filename[:dot_idx] + ('_mul' if backward else '_div') + filename[dot_idx:]

    if backward:
        original_data_filename = filename[:dot_idx - 4] + filename[dot_idx:]
        multipliers_filename = root + 'column_multiplier_' + original_data_filename.split('/')[-1]
    else:
        multipliers_filename = root + 'column_multiplier_' + filename.split('/')[-1]

    if backward:
        # Starts with finding multipliers file
        # multipliers_filename = root + 'column_multiplier_' + filename.split('/')[:-4][-1]

        print(f'Multipliers file found at: {multipliers_filename}')

        multipliers_df = pd.read_csv(multipliers_filename, header=0, sep=sep, dtype=dtype)
        columns = multipliers_df.columns

        print('Columns of original Data and columns of Multipliers correspondingly:')
        print(data.columns, columns)
        print('Original Data Head:')
        print(data.head())
        print('Multipliers Head:')
        print(multipliers_df.head())

        for col in columns:
            data[col] = data[col] * multipliers_df[col].loc['max_std_mean']

        print('New Data Head:')
        print(data.head())

        data.to_csv(new_filename, sep=sep, index=False)
        print(f'File with New Data saved as: {new_filename}')

        original_data_filename = filename[:dot_idx - 4] + filename[dot_idx:]
        original_data = pd.read_csv(original_data_filename, header=0, sep=sep, dtype=dtype)

        print(f'\nOriginal data file found at {original_data_filename}\n')
        print('Comparison of Original Data and Backward Data:')
        sub = data.subtract(original_data)
        print(sub)
        print('\nSum or error by columns:')
        print(sub.sum(axis=0))

    else:
        # Choosing slice that we will calculate mean on
        # _l = min(len(data), 8192)
        _l = len(data)

        # Choosing columns
        columns = [i for i in data.columns if not ignore_time or (ignore_time and i != time_col)]

        multipliers_df = pd.DataFrame(columns=columns)

        # Calculating mean, rounding to 0 decimals
        multipliers_df.loc['general_mean'] = data[columns][:_l].mean().round(0)
        multipliers_df.loc['general_std'] = data[columns][:_l].std().round(0)
        multipliers_df.loc['max_std_mean'] = multipliers_df.max(axis=0)

        # Rounding all the numbers less then 1.5 to 1. as this kind of numbers are already small enough
        multipliers_df[multipliers_df <= 1.5] = 1.

        # Saving multipliers for future immediately
        multipliers_df.to_csv(multipliers_filename, sep=sep, index=False)

        print('Columns of original Data and columns of Multipliers correspondingly:')
        print(data.columns, columns)
        print('Original Data Head:')
        print(data.head())
        print('Multipliers Head:')
        print(multipliers_df.head())
        print(f'File with column multipliers saved as: {multipliers_filename}')

        for col in columns:
            data[col] = data[col] / multipliers_df[col].loc['max_std_mean']

        print('New Data Head:')
        print(data.head())

        print('Distribution after Division:')
        print('Mean: \n', data[columns][:_l].mean())
        print('Std: \n', data[columns][:_l].std())

        data.to_csv(new_filename, sep=sep, index=False)

        print(f'File with New Data saved as: {new_filename}')

    return new_filename


if __name__ == '__main__':
    FILENAME = 'data/BTCBUSD_CandlestickHist.csv'

    FILENAME = column_multiplier(filename=FILENAME,
                                 ignore_time=True,
                                 time_col='Time',
                                 backward=False,
                                 sep=';',
                                 dtype='float64',
                                 )

    print(f'FILENAME {FILENAME}')

    # FILENAME = column_multiplier(filename=FILENAME,
    #                              ignore_time=True,
    #                              time_col='Time',
    #                              backward=True,
    #                              sep=';',
    #                              dtype='float64',
    #                              )

    train_test_split_by_days(filename=FILENAME,
                             train_days=30,
                             test_days=5,
                             window=10,
                             time_step=(60000, 'ms'),
                             to_the_end=True)
