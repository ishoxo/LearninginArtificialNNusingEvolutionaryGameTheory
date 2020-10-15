import pandas as pd
import numpy as np

train_size = 50000
val_size = 10000
test_size = 10000
MNIST_data = pd.read_csv('TSNE_all_untouched')
MNIST_train = MNIST_data[:train_size]
MNIST_val = MNIST_data[train_size:train_size+val_size]
MNIST_test = MNIST_data[train_size+val_size:]


def one_vs_all_data(dataframe, n):
    """

    :param dataframe: MNIST dataframe
    :param n: desired label
    :return: all rows of original dataframe where label is equal to desired label
    """
    dataframe_pos = dataframe.loc[dataframe['label'] == n]
    dataframe_neg = dataframe.loc[dataframe['label'] != n]

    is_n = [1] * len(dataframe_pos)
    is_not_n = [0] * len(dataframe_neg)

    dataframe_pos['new_label'] = is_n
    dataframe_neg['new_label'] = is_not_n

    dataframe_neg = dataframe_neg[:len(dataframe_pos)]
    frames = [dataframe_pos, dataframe_neg]
    d1 = pd.concat(frames)
    d1 = d1.sample(frac=1).reset_index(drop=True)
    d1['new_label'] = d1['new_label'].replace(np.nan, 0)
    d1 = d1.sample(frac=1).reset_index(drop=True)
    return d1
