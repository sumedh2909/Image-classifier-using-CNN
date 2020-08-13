from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
import numpy as np


labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
(_, _),(x_test,y_test)=cifar10.load_data()

x_test=x_test.astype('float32')/255.0
y_test=utils.to_categorical(y_test)

model=load_model('Classifier.h5')

evaluation=model.evaluate(x_test,y_test)
print('Loss is: ',evaluation[0])
print('Accuracy is: ',evaluation[1])


test_image=np.asarray([x_test[0]])

predictn=model.predict(test_image)
category=np.argmax(predictn)

print('Category is :',labels[category])