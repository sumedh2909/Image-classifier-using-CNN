from keras.datasets import cifar10
import keras.utils as utils
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import SGD

labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(x_train, y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0
y_train=utils.to_categorical(y_train)
y_test=utils.to_categorical(y_test)

model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),kernel_constraint=maxnorm(3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=512,kernel_constraint=maxnorm(3),activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer=SGD(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=32)

model.save(filepath='Classifier.h5')



