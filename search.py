from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import random
import os
from keras.models import load_model
from extra_keras_datasets import emnist
from keras.layers import SpatialDropout2D


def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        lr = lr / 2
    return lr


def load_and_preprocess_data():
    (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test, labels_test


def build_model(x):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(x),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.002)),  # Note the regularization
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(x),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.002)),  # Increased regularization
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(x+0.15),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.002)),  # Consistent regularization
        BatchNormalization(),
        Dropout(x+0.25),
        Dense(10, activation='softmax')
    ])
    return model


def evaluate_model(model, x_test, labels_test):
    outputs = model.predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)
    misclassified = np.sum(labels_predicted != labels_test)
    print('Percentage misclassified =', 100 * misclassified / labels_test.size)


def train_model(x):
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    x_train, y_train, x_test, y_test, labels_test = load_and_preprocess_data()
    learning_rate = 0.0012  # Example of a smaller learning rate

    # Create the optimizer with the desired learning rate
    adam_optimizer = Adam(learning_rate=learning_rate)

    net = build_model(x)
    net.compile(loss='categorical_crossentropy', optimizer=adam_optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)

    lr_scheduler = LearningRateScheduler(scheduler)

    history = net.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=80,
        callbacks=[early_stopping, lr_scheduler]  # Add the scheduler here
    )

    net.save("network_for_mnist.h5")
    evaluate_model(net, x_test, labels_test)


def test():
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
    return 100 * correct_classified / labels_test.size


if __name__ == '__main__':
    x = 0.1
    results = []  # List to store the results
    #256
    for i in range(10):
        train_model(x)
        accuracy = test()
        results.append((accuracy, x))  # Store the pair (accuracy, parameter) in the list
        x += 0.05
    # After all loops are over, print the accuracy and parameter one by one
    for accuracy, param in results:
        print(f'Percentage correctly classified EMNIST: {accuracy}% , Parameter: {param}')
    # 0.0012
