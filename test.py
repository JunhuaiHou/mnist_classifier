import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from extra_keras_datasets import emnist


#load .h5 file of arbitrary name for testing (last if more than one)
print(os.getcwd())
for file in os.listdir(os.getcwd()):
    if file.endswith(".h5"):
        print(file)
        net=load_model(file)
net.summary()

#determine what type of network this is
conf=net.layers[0].get_config()
inshape=conf['batch_input_shape']
if inshape[1]==28:
    netType='CNN'
else:
    netType='MLP'


#(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

(input_train, target_train), (x_test, labels_test) = emnist.load_data(type='digits')
x_test = x_test.astype('float32')
x_test /= 255
if netType in ['MLP']:
    x_test = x_test.reshape(x_test.shape[0], -1)
else:
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
correct_classified = sum(labels_predicted == labels_test)
print('Percentage correctly classified EMNIST=', 100 * correct_classified / labels_test.size)
