# Federated learning filtering non-influential data for non-IID data

Federated learning with influence function

Federated learning with ICML 2017 Best paper "influence function" 
This code is compatilble with Tensorflow 2.x which can use eagar execution (from previous TensorFlow 1.x)

![image](https://user-images.githubusercontent.com/45510932/115486336-74e42b00-a291-11eb-8b83-b0e39c32f610.png)


#### 1 versus 7 classification In MNIST dataset problem
------------------------------------------------
below pictures are the most harmful data for federated learning 

![image](https://user-images.githubusercontent.com/45510932/113868137-4de61d80-97ea-11eb-8a2e-e96a28202710.png)


#### After filtering non-influential data
---------------------------------------------
![image](https://user-images.githubusercontent.com/45510932/113869487-c0a3c880-97eb-11eb-838e-6fa21158f7f8.png)


#### How to calculate influence on loss ?

Step 0. Build model
```python3
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True))
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x, y)

```

Step 1. you should calculate s_test (inversed hessian vector product)
```python3
TEST_INDEX = 5

x_test_tf = tf.convert_to_tensor(x_test[TEST_INDEX: TEST_INDEX+1])
y_test_tf = tf.convert_to_tensor(y_test[TEST_INDEX: TEST_INDEX+1])

test_grad_my = grad_z(x_test_tf, y_test_tf, f=model)

```


Step 2. you also should calculate gradient of specific train data (grad z of train data i)
```python3

s_test_my = get_inv_hessian_vector_product(x_train_tf, y_train_tf, test_grad_my, model,
                                            scale=10,
                                            n_recursion=1000,
                                            verbose=False)
                                            
```

Step 3. you should multipliy gradient of train with s_test (for each train data)
```python3

for i in range(train_sample_num):
    
    # Get train grad
    train_grad = grad_z(x_train_tf[i: i+1], y_train[i: i+1], model, for_train=True)
    loss_diff_approx[i] = multiply_for_influe(train_grad, s_test_my) / train_sample_num
    
```


-------------------------------------------------
# Estimation of loss in model without bias

<img width="442" alt="Screen Shot 2021-04-07 at 9 44 28 AM" src="https://user-images.githubusercontent.com/45510932/113794629-f2824400-9785-11eb-88de-3103f213596f.png">

# Estimation of loss in model with bias

<img width="452" alt="Screen Shot 2021-04-07 at 9 44 34 AM" src="https://user-images.githubusercontent.com/45510932/113794637-f44c0780-9785-11eb-8ccd-90d74d15f9c0.png">
-------------------------------------------------


# Calculation
----------------------------------
![image](https://user-images.githubusercontent.com/45510932/113869112-51c66f80-97eb-11eb-8994-5f19e27e7496.png)

![image](https://user-images.githubusercontent.com/45510932/113869142-57bc5080-97eb-11eb-9c39-b6318d15eb7b.png)

![image](https://user-images.githubusercontent.com/45510932/113869197-660a6c80-97eb-11eb-93b4-f4f20f1b30a1.png)




#### Requirement
tensorflow 2.x

numpy 1.xx (we recommand to use 1.18.5 which we used in our paper)

sklearn 0.23.x (only used in simulation)


#### Install
```git clone https://github.com/4pygmalion/Federated_learning-filtering-non-influence-data.git
```


This code was inspired by 
https://github.com/nayopu/influence_function_with_lissa

and ICML 2017 best paper Understanding Black-box Predictions via Influence Functions (https://arxiv.org/pdf/1703.04730)
