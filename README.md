# Federated learning filtering non-influential data for non-IID data

Federated learning with influence function

Federated learning with ICML 2017 Best paper "influence function" 
This code is compatilble with Tensorflow 2.x which can use eagar execution (from previous TensorFlow 1.x)


#### 1 versus 7 classification In MNIST dataset problem
------------------------------------------------
below pictures are the most harmful data for federated learning 

![image](https://user-images.githubusercontent.com/45510932/113868137-4de61d80-97ea-11eb-8a2e-e96a28202710.png)


#### After filtering non-influential data
---------------------------------------------
![image](https://user-images.githubusercontent.com/45510932/113869487-c0a3c880-97eb-11eb-838e-6fa21158f7f8.png)



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



This code was inspired by 
https://github.com/nayopu/influence_function_with_lissa
