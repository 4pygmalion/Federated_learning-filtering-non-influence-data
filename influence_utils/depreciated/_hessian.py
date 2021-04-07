import numpy as np
import tensorflow as tf
from ._etc import calc_loss

def hessian_loss(xs, ys ,f):
    ''' To get hessian matrix of loss function in training data

    Parameters
    ----------
    xs: tf.Tensor. in training dataset
    ys: tf.Tensor. in training dataset
    f: tf.keras.models.Sequntial()

    Returns
    -------
    hessian matrix of loss function: list. including hessian matrix w.r.t each parameters
    '''
    Hs = []
    for ws in f.trainable_weights:
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xs)
            tape2.watch(ws)

            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xs)
                tape1.watch(ws)

                ys_hat = f(xs)
                loss = tf.keras.losses.binary_crossentropy(ys, ys_hat)

            grads = tape1.gradient(loss, ws)
        
        H = tape2.jacobian(grads, ws)


        # Reshaping
        t_shape = H.shape[0] * H.shape[1]
        H = tf.reshape(H, shape=(t_shape, t_shape)) # Reshaping into  (m by m shape) for inversed hessian
        Hs.append(H)

    return Hs


def hvp(xs, ys, v, model):
    '''
    1. get loss, instead of loss
    2. get w from model, instread of parameters

    Parameters
    ----------
    xs: tf.tensor or tf.Varaible.
        in train data
    ys: tf.tensor or tf.Variable

    v: grad z with respect to all test point
        (e.g h_estimate)
        (e.g. v = grad_z(z_test, t_test, model, gpu))

    model: tf.keras.model
        (originally, h_estimation)

    Return
    ------
    hvp: list of tf.tensor, containing product of hessian and v
    '''

    w = model.trainable_weights

    if len(w) != len(v):
        raise (ValueError("w and v must have the same length."))

    # Second derivate with elementwise products
    with tf.GradientTape() as tape1:
        tape1.watch(xs)

        # 1. First partial gradient
        with tf.GradientTape() as tape2:
            tape2.watch(xs)
            y_hat = model(xs)
            loss = calc_loss(ys, y_hat)
        first_grad = tape2.gradient(loss, w)

        # 2. Elementwise vector product
        elementwise_products = 0
        for grad_, v_ in zip(first_grad, v):
            elementwise_products += tf.reduce_sum(grad_ * v_)

    # 3. Second partial
    grads_with_none = tape1.gradient(elementwise_products, w)


    return_grads = [grad_elem if grad_elem is not None else tf.zeros(shape=(v[-1].shape))
                    for grad_elem in grads_with_none]

    return return_grads


