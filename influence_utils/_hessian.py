# inspired by
# https://ryokamoi.github.io/blog/tech/2020/06/18/influence-functions.html
# https://github.com/ryokamoi/test_pytorch_influence_functions/blob/master/model.py

import numpy as np
import tensorflow as tf


def grad_z(x, y, f, for_train=False, decay_rate=0.01):
    ''' Calculates the gradient of data points z for recursively computing HVP

    (verified)

    Parameters
    ----------
    x: tf.Tensor, a training data point
    y: tf.Tensor, label of a training data point
        e.g) as image. (batch, 3, 128, 128)
        e.g) as static tabular data. (batch, features)
        e.g) as timeseries data. (batch, time, features)

    f: tensorflow.keras.Model

    for_train: bool, whether use regularization or not.
        default) False

    decay_rate: float, the weight of regularization panelty


    Return
    ------
    grad_z: list of tf.tensor
    '''

    if len(x.shape) <= 1:
        x = tf.expand_dims(x, axis=0)
    if len(y.shape) <= 1:
        y = tf.expand_dims(y, axis=0)

    if for_train:
        with tf.GradientTape() as tape:
            ws = f.trainable_weights
            tape.watch(ws)
            tape.watch(x)
            y_hat = f(x)
            loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_hat))
            panelty = tf.math.reduce_sum([tf.nn.l2_loss(w) for w in ws]) * decay_rate
            loss = loss + panelty
    else:
        with tf.GradientTape() as tape:
            ws = f.trainable_weights
            tape.watch(ws)
            tape.watch(x)
            y_hat = f(x)
            loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_hat))

    return tape.gradient(loss, ws)


def get_hessian_vector_product(x_samples, y_samples, f, test_grad):
    ''' Caculate the product of hessian and implicit vector product (Hv)
    Not inversed hessian matrix


    Pramaters
    ---------
    x_samples: tf.Tensor, featrues of train datapoint from sampling z_s_test
    y_samples: tf.Tensor, labels of train datapoint from sampling z_s_test
    f: tf.keras.model.Sequntial, Model
    test_grad: list of tf.Tensor
        e.g) v = grad_z(z_test, t_test, model). our code require list type as input


    Returns
    ------
    hessian vector product(HVP) : list of tf.Tensor
    '''

    # Input validity: 1) tf.Tensor, 2) same shape with first_grads
    if isinstance(test_grad, list):
        for elem in test_grad:
            if not isinstance(elem, tf.Tensor):
                raise ValueError('''Test grad must be tf.Tensor type''')

    with tf.GradientTape() as tape1:
        ws = f.trainable_weights
        tape1.watch(ws)
        first_grads = grad_z(x_samples, y_samples, f, for_train=True)
        tape1.watch(first_grads)

        if isinstance(test_grad, list) and isinstance(first_grads, list):
            for grad_t, grad_s_t in zip(test_grad, first_grads):
                assert grad_t.shape == grad_s_t.shape, "expected grad(s_t) and grad(t) have same shape but got{} {}".format(
                    grad_t.shape,
                    grad_s_t.shape)

        grads = [_fist_grad * _test_grad for _fist_grad, _test_grad in zip(first_grads, test_grad)]

    return tape1.gradient(grads, ws)


def get_inv_hessian_vector_product(x_train, y_train,
                                   v,
                                   f,
                                   num_samples=5,
                                   n_recursion=1000,
                                   damp=0.01,
                                   scale=10,
                                   verbose=True
                                   ):
    '''
    Parameters
    ----------
    x_train: tf.Tensor, or np.array.
        entire training dataset
    y_train: tf.Tensor, or  np.array.
        entrie training dataset
    v: test_grad.
    f: tf.keras.model.Sequntial or Model
    n_recursion: int. number of recursive loop

    Returns
    -------
    inversed hessian-vector product : list
    

    Reference
    ---------
    # https://github.com/nayopu/influence_function_with_lissa/blob/master/model.py#L54
    # https://github.com/nimarb/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_function.py
    '''

    inverse_hvp = None
    n_data = len(x_train)

    for i in range(num_samples):
        if verbose: print('Sample iteration [{}/{}]'.format(i + 1, num_samples))

        cur_estimate = v
        np.random.seed(42)
        permuted_indices = np.random.permutation(range(n_data))

        for j in range(n_recursion):
            x_sample = x_train[permuted_indices[j]:permuted_indices[j] + 1]
            y_sample = y_train[permuted_indices[j]:permuted_indices[j] + 1]

            # get hessian vector product
            hvp = get_hessian_vector_product(x_sample, y_sample, f, cur_estimate)

            cur_estimate = [_v + _cur - _hvp / scale for _v, _cur, _hvp in zip(v, cur_estimate, hvp)]

        if inverse_hvp is None:
            inverse_hvp = [_cur / scale for _cur in cur_estimate]
        else:
            inverse_hvp = [_inv_hvp + _cur / scale for _inv_hvp, _cur in zip(inverse_hvp, cur_estimate)]

    inverse_hvp = [_inv_hvp / num_samples for _inv_hvp in inverse_hvp]
    return inverse_hvp