import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

data=pd.read_csv('Data/Data.csv').astype('float32')
#print(data.head(10))

X = data.drop('0',axis = 1)
y = data['0']

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.2)

x_train=np.reshape(x_train.values, (x_train.shape[0], 28,28,1)).astype('float32')
x_test=np.reshape(x_test.values, (x_test.shape[0], 28,28,1)).astype('float32')

y_train=tf.keras.utils.to_categorical(y_train,num_classes=26)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=26)

#print(x_train.shape)
#print(x_test.shape)

model=Sequential()

model.add(Conv2D(32,kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(26,activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_train,
          epochs=3,
          verbose=1,
          validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('Model/model.h5')