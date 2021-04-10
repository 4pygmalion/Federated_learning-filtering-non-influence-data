# import numpy as np
import tensorflow as tf
import tqdm
import os


from ._etc import calc_loss, save_pickle
from ._hessian import hvp, hessian_loss
#
#
# def _search_kernel_weight(weights, weight_name='kernel'):
#     '''Serach kernel weight
#
#     Parameters
#     ---------
#     weights: tensorflow.Tensor
#     name: string
#
#     Return
#     -----
#     Weights or None
#     '''
#
#     import re
#     mask = bool(re.search(weight_name, weights.name))
#     if mask:
#         return weights
#     else:
#         return None


def grad_zs(z, targets, model):
    ''' Calculates the gradient of data points z for recursively computing HVP

    Parameters
    ----------
    z: tf.Tensor, batches in training data point
        e.g) as image. (batch, 3, 128, 128)
        e.g) as static tabular data. (batch, features)
        e.g) as timeseries data. (batch, time, features)

    targets: tf.Tensor. label data
    model: tensorflow.keras.Model

    Return
    ------
    grad_z: list of tf.tensor
    '''

    # if z.shape[0] <= 1:
    #     raise ValueError('Tensor z must be a t data points. But current z is:', z.shape)
    if len(z.shape) <= 1:
        z = tf.expand_dims(z, axis=0)


    with tf.GradientTape() as tape:
        # recording gradient
        tape.watch(z)

        # Get loss
        y = model(z)
        loss = calc_loss(y, targets)

        # get model parameters
        ws = model.trainable_weights
        # ws = [_search_kernel_weight(w) for w in ws if _search_kernel_weight(w) is not None]
    return tape.gradient(loss, ws)

def save_s_test(z_test, t_test, model, training_data,
                damp=0.01,
                scale=25.0,
                recursion_depth=500,
                save=True,
                filename='client_h'
                ):
    ''' save_s_test for each test data
    
    Parameters
    ----------
    z_test: 
    t_test
    model
    training_data
    damp
    scale
    recursion_depth
    save
    filename

    Returns
    -------
    '''

    i = 0
    for z_test in zip(z_test, t_test):
        v = grad_zs(z_test[0], t_test[1], model)  # a data point in test set
        h_init_estimates = v.copy()

        for i in range(recursion_depth):
            j = 0
            for x, y in zip(*training_data):  # w.r.t training data
                x = tf.expand_dims(x, axis=0)  # x shape (1, features)<- x shape: (features,)
                y_hat = model(x)
                hv = hvp(x, y, h_init_estimates, model)

                h_estimate = [_v + (1 - damp) * h_estimate - _hv / scale
                              for _v, h_estimate, _hv in zip(v, h_init_estimates, hv)]
                j += 1
            print('Recursive process [{}]th processed'.format(i), end='\r')


        if not os.path.exists('./s_test/'):
            os.mkdir('./s_test/')
        filename_ = filename + '_{}'.format(i)
        save_pickle(h_estimate, './s_test/{}'.format(filename_))
        i+=1

def s_test(z_test, t_test, model, training_data,
           damp=0.01,
           scale=25.0,
           recursion_depth=500,
           save=True,
           filename='client_h'):

    '''Get "inversed Hessian * Vector Product (H^-1 * V)

    step 1. uniformly sample t points from training data
    step 2. define inv(H0) = v
    step 3. recursively compute
    step 4. stop iteration

    Parameters
    ----------
    z_test: tf.Tensor, data point
    t_test: tf.Tensor, data point
    model: tf.tensorflow.keras.Model
    training_data: tuple. (x, y)

    Return
    ------
    h_estimate

    '''

    v = grad_zs(z_test, t_test, model)  # a data point in test set
    h_init_estimates = v.copy()

    for i in range(recursion_depth):
        j = 0
        for x, y in zip(*training_data): # w.r.t training data
            x = tf.expand_dims(x, axis=0) # x shape (1, features)<- x shape: (features,)
            y_hat = model(x)
            hv = hvp(x, y, h_init_estimates, model)

            h_estimate = [_v + (1 - damp) * h_estimate - _hv / scale
                          for _v, h_estimate, _hv in zip(v, h_init_estimates, hv)]
            # break
            j+=1
            # print('Hessian estimation for each {}th data'.format(j), end='\r')
        print('Recursive process [{}]th processed'.format(i), end='\r')

    if save:
        if not os.path.exists('./s_test/'):
            os.mkdir('./s_test/')
        # np.save(h_estimate, './s_test/{}'.format(filename))
        save_pickle(h_estimate, './s_test/{}'.format(filename))
    return h_estimate


def calc_influence_loss_upweighting(z, test_data, model,
                               inv_hessian=None):
    '''
    Parameters
    ----------
    z: tuple. a single data point of training set
    test_data: tuple (x_test, y_test)

    '''
    if not isinstance(test_data, tuple):
        raise ValueError('Test data is not tuple. it is ', type(test_data))
    if not isinstance(z, tuple):
        raise ValueError('Test data is not tuple. it is ', type(test_data))


    if inv_hessian is None:
        h = hessian_model_wrt_x(test_data[0], model)
        inv_h = tf.linalg.inv(h)

    # Grad loss of z_test
    x_test, y_test = test_data
    grad_z_test = grad_zs(x_test, y_test, model)

    # Grad loss of data point z
    x_point, y_point = z
    grad_z_point = grad_zs(x_point, y_point, model)

    return grad_z_test, grad_z_point




def calc_influence_single(model, training_data, test_data,
                          test_id_num,
                          s_test_vec=None,
                          recursion_depth=50):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        training_data: tuple(x_train, y_train)
        test_data: tuple(x_test, y_test)
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""

    # if s_test vec were not given:
    if s_test_vec is None:
        z_tests, t_tests = test_data
        z_, t_ = z_tests[test_id_num], t_tests[test_id_num]
        print('s_test vec will be generated')
        s_test_vec = s_test(z_test=z_, t_test=t_, model=model, training_data=training_data)
    print('S test vector was generated')

    # Load train dataset
    xs, ys = training_data
    n_train_data = len(xs)

    # Calculate the influence function
    influences = []
    print('Calculate influence function with each data point')

    for i in range(n_train_data):

        z, t = xs[i], ys[i]
        z = tf.expand_dims(z, axis=0)
        t = tf.expand_dims(t, axis=0)

        grad_z_vec = grad_zs(z, t, model)
        return grad_z_vec
        tmp_influences = -sum([tf.math.reduce_sum(k*j) for k, j in zip(grad_z_vec, s_test_vec)]) # s_test * L(z, theta)])
        influence = (tmp_influences / n_train_data)
        influences.append(influence)

        if i % 1000 == 0:
            print('Process i complete {}/{}'.format(i, n_train_data), end='\r')
    return influences