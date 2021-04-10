import numpy as np


def influe_groupby_label(data, y_train, y_val):
    '''
    data: np.arary
        n train x n val

    return
    ------
    influes: np.ndrrary.
        n train
    '''
    if len(data) != len(y_train):
        raise ValueError('the length of data and label must have same length')

    influe_groupby = np.empty(shape=(len(data)))

    for i in range(len(influe_groupby)):
        my_label = y_train[i]
        same_label_idx = np.where(my_label == y_val)[0]
        avg_influe = data[i, same_label_idx].mean()
        influe_groupby[i] = avg_influe

    return influe_groupby


def filtering_influce(data_pool, influe_master, alpha):
    '''

    Parameters
    ----------
    data_pool: dict

    influe_master: dict
        key: client_name
        value: np.array (m, n)
    alpha: float.
        percentile

    Returns
    -------
    filtered data_poot: dcit


    '''



    # for each client
    client_names = data_pool.keys()

    _data_pool = dict()
    for client_name in client_names:
        c_influe = influe_master[client_name]

        mask1 = c_influe > 0
        mask2 = c_influe > np.percentile(c_influe, alpha)
        mask = mask1 | mask2

        # replacement
        x = data_pool[client_name]['train'][0][mask]
        y = data_pool[client_name]['train'][1][mask]

        _data_pool[client_name] = {'train': [x, y]}

    return _data_pool