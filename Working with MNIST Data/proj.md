```python
from __future__ import print_function 
import keras
```


```python
#import all nedded function
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense,Dropout,Flatten 
from keras.layers import Conv2D,MaxPool2D 
from keras import backend as k 
```


```python
#preparing data 
#we will divid the traning data set to a batches 
#not all data will be trained at once 
#this will increase the efficency of the model 
batch_size=128 
number_of_classes = 10 # depending on the training data classification 
#epochs >> i will train the data in forward and backward way only one time 
epochs=10  # i will go forwardand backowd a 10 time

```


```python
img_row,img_col=28,28
(x_train,y_train),(x_test,y_test)=mnist.load_data()
```


```python
#ensuring that the data we have are in correct format (Gray_scale)
# k.image_data_format() retun A string, either 'channels_first' or 'channels_last'
if k.image_data_format()=='channels_first':
    x_train=x_train.reshape(x_train.shape[0],1,img_row,img_col)
    x_test=x_test.reshape(x_test.shape[0],1,img_row,img_col)
    input_shape=(1,img_row,img_col)
else : 
    x_train=x_train.reshape(x_train.shape[0],img_row,img_col,1)
    x_test=x_test.reshape(x_test.shape[0],img_row,img_col,1)
    input_shape=(img_row,img_col,1) 


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255 
print("\n traing shape",x_train.shape)

#we need to conver all classes  to a binary classed formate 
y_train = keras.utils.to_categorical(y_train,number_of_classes) 
y_test = keras.utils.to_categorical(y_test,number_of_classes)

#after every CNN layer we need to add a max pooling layer that will chisce the max result of the convolution calc
model=Sequential() 
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()


```

    
     traing shape (60000, 28, 28, 1)
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_3 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 128)               1179776   
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss=keras.losses.categorical_crossentropy
              ,optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test,y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/10
    60000/60000 [==============================] - 160s 3ms/step - loss: 0.2006 - accuracy: 0.9387 - val_loss: 0.0643 - val_accuracy: 0.9798
    Epoch 2/10
    60000/60000 [==============================] - 160s 3ms/step - loss: 0.0471 - accuracy: 0.9860 - val_loss: 0.0375 - val_accuracy: 0.9871
    Epoch 3/10
    60000/60000 [==============================] - 159s 3ms/step - loss: 0.0294 - accuracy: 0.9908 - val_loss: 0.0340 - val_accuracy: 0.9882
    Epoch 4/10
    60000/60000 [==============================] - 161s 3ms/step - loss: 0.0202 - accuracy: 0.9937 - val_loss: 0.0333 - val_accuracy: 0.9892
    Epoch 5/10
    60000/60000 [==============================] - 191s 3ms/step - loss: 0.0130 - accuracy: 0.9959 - val_loss: 0.0368 - val_accuracy: 0.9894
    Epoch 6/10
    60000/60000 [==============================] - 177s 3ms/step - loss: 0.0091 - accuracy: 0.9974 - val_loss: 0.0326 - val_accuracy: 0.9900
    Epoch 7/10
    60000/60000 [==============================] - 173s 3ms/step - loss: 0.0058 - accuracy: 0.9984 - val_loss: 0.0296 - val_accuracy: 0.9909
    Epoch 8/10
    60000/60000 [==============================] - 173s 3ms/step - loss: 0.0042 - accuracy: 0.9989 - val_loss: 0.0378 - val_accuracy: 0.9894
    Epoch 9/10
    60000/60000 [==============================] - 166s 3ms/step - loss: 0.0027 - accuracy: 0.9993 - val_loss: 0.0395 - val_accuracy: 0.9903
    Epoch 10/10
    60000/60000 [==============================] - 167s 3ms/step - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0383 - val_accuracy: 0.9910
    




    <keras.callbacks.callbacks.History at 0x22e30efe788>




```python
score=model.evaluate(x_test,y_test,verbose=0)
print("test loss",score[0])
print("test acc",score[1])

```

    test loss 0.038334573550052965
    test acc 0.9909999966621399
    


```python
model.save("model.h5")
print("Saved")
```

    Saved
    


```python

```


```python

```
