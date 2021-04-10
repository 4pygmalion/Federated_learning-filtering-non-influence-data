from ._hessian import get_inv_hessian_vector_product
from ._hessian import grad_z

import numpy as np
import tensorflow as tf


def multiply_for_influe(a, b):
    '''function to multiply (@), by avoiding error tf.transpose to bias (scalar)

    Parameters
    ----------
    a, b: tf.Tensor

    Returns
    -------

    '''

    if len(a) != len(b):
        raise ValueError('Tensor a, b must have same length.')

    for i in range(len(a)):
        if a[i].shape != b[i].shape:
            raise ValueError('Tensor a, b must have same shape.')

    result = []
    for i in range(len(a)):
        t1, t2 = a[i], b[i]

        if len(t1.shape) == 1:
            result.append(t1 * t2)
        else:
            result.append(tf.transpose(t1) @ t2)

    return sum([t.numpy().item() for t in result])



# def cal_influence_data(x_train, y_train, x_test, y_test, train_data, f, verbose=True):
#     '''
#     Parameters
#     ----------
#     x_train: tf.Tensor or np.array.
#         features
#     y_train: tf.Tensor or np.array.
#         label
#     train_data: list.
#         entire training dataset (xs, ys)
#     test_data: list.
#         a data point (x, y)
#     f: tf.keras.model.Sequntial, Model
#
#
#     Return
#     ------
#     influnece value: float32
#     '''
#
#     # grad_z
#     grad_z_train = grad_z(x_train, y_train, f)
#
#     # get inversed hessian vector product
#     grad_z_test = grad_z(x_test, y_test, f)  # as v. must be loss gradient of single test data point
#     inv_Hv = get_inv_hessian_vector_product(x_test, y_test, train_data, f, verbose=verbose, n_recursion=300)
#
#     return -sum([tf.math.reduce_sum(k *j) for k, j in zip(grad_z_train, inv_Hv)]).numpy() / x_train.shape[0]
#
#
# def avg_upweighting_influence_on_loss_by_class(x_train, y_train, train_data, test_data, f, verbose=True):
#     '''
#
#     Parameters
#     ----------
#     x_train: tf.Tensor. a train data
#     y_train: tf.Tensor. a train data
#     train_data: list. entire train data [xs, ys]
#     test_data: list. entire test data [xs, ys]
#     f:  tf.keras.model.Sequntial, Model
#     verbose: bool
#
#     Returns
#     -------
#     average influence on test data set which have same class of train data.
#     '''
#
#
#     if len(y_train.shape) != 1:
#         raise ValueError('expected y_train must be 1 dimensional, but {} was input'.format(len(y_train.shape)))
#
#
#     # Get test data which have same label of train data
#     xs_test, ys_test = test_data[0].numpy(), test_data[1].numpy()
#     idx = np.where(ys_test == y_train)[0]
#     test_data = [tf.convert_to_tensor(xs_test[idx]), tf.convert_to_tensor(ys_test[idx])]
#
#     # Main
#     i = 0
#     n_loop = len(test_data[0])
#     influes = []
#
#     for x_test, y_test in zip(*test_data):
#         influ = cal_influence_data(x_train, y_train, train_data, [x_test, y_test], f, verbose=False)
#         influes.append(influ)
#         i += 1
#
#         # Versbose
#         if verbose:
#             if i % 10 == 0:
#                 print('Number of loop (influence function): {} / {}'.format(i, n_loop), end='\r')
#
#     influ_mean = tf.reduce_mean(influes)
#     return influ_mean.numpy()
#
#
#
#
#
# def avg_upweight_inluence_on_loss(x_train, y_train, train_data, test_data, f, verbose=True):
#     ''' Calculate influence of a specific x_train, y_train data on overall test data
#
#     THis function will be depreciate because of class adjusting
#
#     Parameters
#     ----------
#     x_train: tf.Tensor or np.array.
#         features
#     y_train: tf.Tensor or np.array.
#         label
#     train_data: list.
#         entire training dataset
#     test_data: list.
#         entire test data
#     f: tf.keras.model.Sequntial, Model
#
#     return
#     ------
#     average influence on test data set.
#     '''
#
#     i = 0
#     n_loop = len(test_data[0])
#     influes = []
#
#     for x_test, y_test in zip(*test_data):
#         influ = cal_influence_data(x_train, y_train, train_data, [x_test, y_test], f, verbose=False)
#         influes.append(influ)
#         i += 1
#
#         # Versbose
#         if verbose:
#             if i % 10 == 0:
#                 print('Number of loop (influence function): {} / {}'.format(i, n_loop), end='\r')
#
#     influ_mean = tf.reduce_mean(influes)
#     return influ_mean.numpy()