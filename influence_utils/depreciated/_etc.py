import tensorflow as tf
import pickle

def calc_loss(ys, targets, use_proba=False, reduce=False):
    '''Calcualtes the loss model (negative log-likelhood)
    the negative log-likelihood is the log loss, or (binary) cross-entropy for (binary) classification problems

    Paramters
    ---------
        ys: tf.tensor, input with size (batch, number of classes)
        targets = tf.tensor, shape=(N, n_class) with double n_class

    Return
    ------
        loss: scalar
    '''
    # if not isinstance(targets, tf.Tensor):
    #     raise TypeError('Expected target is tf.Tensor But target is', type(targets))
    # if not (targets.dtype == tf.int64) or not (ys.dtype == tf.int64):
    #     raise TypeError('Expected target and ys were tf.Tensor. But target and ys are',
    #                     type(targets.dtype), type(ys.dtype))


    if use_proba:
        bools = (ys > 0.5)
        ys = tf.cast(bools, dtype='float32')
        targets = tf.cast(targets, dtype='float32')

    if reduce:
        cce = tf.keras.losses.BinaryCrossentropy()
    else:
        cce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    # print('ys:', ys, 'target:', targets)
    loss = cce(ys, targets)

    return loss





def save_pickle(object_name, file_name):
    '''
    Parameters
    ----------
    object_name: python object
    file_name: str. target path of file


    Example
    -------
    None
    '''
    with open(file_name, 'wb') as handle:
        pickle.dump(object_name, handle, protocol=pickle.HIGHEST_PROTOCOL)