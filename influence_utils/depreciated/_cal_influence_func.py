from ._hessian import hessian_loss
from ._hessian import hvp
from ._influence_func import grad_zs, s_test
from ._etc import calc_loss, save_pickle

import tensorflow as tf
import numpy as np
import os
import re



class InfluenceExplorer(object):

    def __init__(self, model):
        self.model = model

    def _load_pickle(self, file_name):
        '''
        Parameters
        ----------
        filename: str. file path


        Return
        ------
        python object

        '''
        import pickle
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)

    def _load_s_test(self, client_name):
        '''

        Parameters
        ----------
        client_name: str

        Returns
        -------
        s_test. pickle

        Raises
        ------
        IOError: file not existed
        '''
        files = os.listdir('./s_test')
        s_test = [file for file in files if bool(re.search(client_name, file))][0]

        pckl = self._load_pickle('./s_test/'+s_test)
        return pckl


    def grad_z(self, z_test, t_test):
        v = grad_zs(z_test, t_test, self.model)
        return v


    def get_inv_H(self, z_test, t_test, client_name):
        def replace_nan(tensor):
            arr = tensor.numpy()
            idx = np.where(np.isnan(arr))

            arr[idx] = 0

            return tf.convert_to_tensor(arr)
        
        # load s test
        s_test = self._load_s_test(client_name)

        # calculate v
        v = self.grad_z(z_test, t_test)
        inv_H = [k/j for k, j in zip(s_test, v)]

        # repalce nan / inf
        inv_H = [replace_nan(h) for h in inv_H]
        inv_H = [tf.clip_by_value(h, clip_value_min=-1e+5, clip_value_max=1e+5) for h in inv_H]

        return inv_H

    def save_s_test(self, z_test, t_test, model, training_data,
                    # damp=0.01,
                    # scale=25.0,
                    t=500,
                    recursion_depth=500,
                    filename='client_h'
                    ):
        ''' save_s_test for each test data

        Parameters
        ----------
        z_test: np.array, feature: is test set (in code; a datapoint in paper)
        t_test: np.array, label: is test set
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

        v = grad_zs(z_test, t_test, model)  # a data point in test set
        h_estimate = v.copy()

        i=1
        for _ in range(recursion_depth):
            xs, ys = training_data[0][:t*i], training_data[1][:t*i]
            grad_z_wrt_zs = grad_zs(xs, ys, model)

            h_estimate = [_v + (1 - _grad_z_wrt_zs) * _hv
                         for _v, _grad_z_wrt_zs, _hv in zip(v, grad_z_wrt_zs, h_estimate)]

            # for x, y in zip(*training_data):  # w.r.t training data
            #     x = tf.expand_dims(x, axis=0)  # x shape (1, features)<- x shape: (features,)
            #     hv = hvp(x, y, h_estimate, model)
            #     return hv
            #     h_estimate = [_v + (1 - _h_estimate) - _hv / scale
            #                   for _v, _h_estimate, _hv in zip(v, h_estimate, hv)]

            print('Recursive process [{}]th processed'.format(_), end='\r')

        if not os.path.exists('./s_test/'):
            os.mkdir('./s_test/')

        filename_ = './s_test/' + filename + '_r_{}'.format(str(_+1))
        save_pickle(h_estimate, filename_)
        return h_estimate


    def cal_grad_z_mean(self, test_data):
        # 2.Grad_z of test set
        grad_z_test_list = [self.grad_z(x, y) for x, y in zip(*test_data)]
        v1 = [_[0] for _ in grad_z_test_list]
        v2 = [_[1] for _ in grad_z_test_list]

        grad_z_test_mean = [tf.math.reduce_mean(v1, axis=0), tf.math.reduce_mean(v2, axis=0)]
        grad_z_test_mean_t = [tf.transpose(g) for g in grad_z_test_mean]
        return grad_z_test_mean_t


    def cal_influence_up_loss(self,
                              train_data,
                              inv_H,
                              test_data=None,
                              grad_z_mean=None
                              ):
        '''

        Parameters
        ----------
        train_data: list.
            x, y of a data point in training data
        test_data: list.
            x, y of all test data points
        model: list

        Returns
        -------
        '''

        # Data validty
        if (test_data is None) & (grad_z_mean is None):
            raise ValueError('test_data or grad_z_mean must be input at least one')

        # 2.Grad_z of test set
        if grad_z_mean is None:
            grad_z_test_list = [self.grad_z(x, y) for x, y in zip(*test_data)]
            v1 = [_[0] for _ in grad_z_test_list]
            v2 = [_[1] for _ in grad_z_test_list]

            grad_z_test_mean = [tf.math.reduce_mean(v1, axis=0), tf.math.reduce_mean(v2, axis=0)]
            grad_z_test_mean_t = [tf.transpose(g) for g in grad_z_test_mean]

        # 1.Grad_z of training data point
        x_train, y_train = train_data[0], train_data[1]
        grad_z_train = self.grad_z(x_train, y_train)

        # 2.Hessian
        inv_H = inv_H

        # 3. Calculate influence function with upweighting, loss
        _influ =  [i @ j @ tf.transpose(k)
                     for i, j, k in zip(grad_z_mean, inv_H, grad_z_train)]
        _influ = sum([tf.math.reduce_sum(m) for m in _influ])

        return _influ.numpy()


def cal_influence_up_loss(z, train_data, test_data, model,
                          damp=0.01,
                          scale=25.0,
                          n_recursion=3000):
    '''

    Parameters
    ----------
    z: list.
        a data point. including x, y of data point in training data
    train_data: list.
        including x, y of test data points [xs, ys]
    test_data: list.
        including x, y of test data points [xs, ys]
    model: tf.keras.models.Model or Sequntial

    Returns
    -------
    influcen. float32
    '''

    # 1. s_test
    x, y = z[0], z[1]
    s_test_value = s_test(x, y, model, train_data, damp, scale, n_recursion)

    # 2. grad_z
    grad_z = grad_zs(x, y, model)

    # influnce function
    influ = -sum([tf.math.reduce_sum(k * j).numpy() for k, j in zip(s_test_value, grad_z)])

    return influ




def save_s_test(z_test, t_test, model, training_data,
                damp=0.01,
                scale=25.0,
                recursion_depth=500,
                filename='client_h'
                ):
    ''' save_s_test for each test data

    Parameters
    ----------
    z_test: np.array, feature: is test set
    t_test: np.array, label: is test set
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

    v = grad_zs(z_test, t_test, model)  # a data point in test set
    h_init_estimates = v.copy()

    for _ in range(recursion_depth):
        for x, y in zip(*training_data):  # w.r.t training data
            x = tf.expand_dims(x, axis=0)  # x shape (1, features)<- x shape: (features,)
            hv = hvp(x, y, h_init_estimates, model)

            h_estimate = [_v + (1 - damp) * h_estimate - _hv / scale
                          for _v, h_estimate, _hv in zip(v, h_init_estimates, hv)]

        print('Recursive process [{}]th processed'.format(_), end='\r')

    if not os.path.exists('./s_test/'):
        os.mkdir('./s_test/')

    filename_ = './s_test/' + filename + '_r_{}'.format(str(_))
    save_pickle(h_estimate, filename_)






    def influence_up_loss(self, train_data, test_data, inv_H):
        '''

        Parameters
        ----------
        z: list. including x, y of data point in training data
        test_data: list. including x, y of test data points
        model: tf.keras.models.Model or Sequntial

        Returns
        -------

        '''
        layers = model.layers
        ws = [l.get_weights() for l in layers]

        # 1.Grad_z of training data point
        x_train, y_train = z[0], z[1]  # in training
        grad_z_train = grad_zs(x_train, y_train, model)[0]

        # 2.Grad_z of test set
        grad_z_test_list = [grad_zs(x, y, model) for x, y in zip(*test_data)]
        grad_z_test_mean = tf.math.reduce_mean(grad_z_test_list, axis=0)
        grad_z_test_mean =  tf.reshape(grad_z_test_mean, shape=grad_z_train.shape)

        # 3.Hessian
        H = hessian_loss(train_data[0], train_data[1], model)[0]
        inv_H = tf.linalg.inv(H)

        # Calculate influence function with upweighting, loss
        influnece =  tf.transpose(grad_z_test_mean) @ inv_H @ grad_z_train
        return influnece.numpy()