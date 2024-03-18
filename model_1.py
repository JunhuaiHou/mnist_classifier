from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import random
from extra_keras_datasets import emnist


def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        lr = lr / 2
    return lr


def load_and_preprocess_data():
    # Load MNIST data
    (x_train_mnist, labels_train_mnist), (x_test_mnist, labels_test_mnist) = mnist.load_data()
    # Load EMNIST data - ensure you have a similar function available
    (x_train_emnist, labels_train_emnist), (x_test_emnist, labels_test_emnist) = emnist.load_data(type='digits')

    # Combine datasets
    x_train = np.concatenate([x_train_mnist, x_train_emnist], axis=0)
    labels_train = np.concatenate([labels_train_mnist, labels_train_emnist], axis=0)
    x_test = np.concatenate([x_test_mnist, x_test_emnist], axis=0)
    labels_test = np.concatenate([labels_test_mnist, labels_test_emnist], axis=0)


    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(labels_train, 10)
    y_test = to_categorical(labels_test, 10)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test, labels_test


def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.002)),  # Note the regularization
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.002)),  # Increased regularization
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.35),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.002)),  # Consistent regularization
        BatchNormalization(),
        Dropout(0.45),
        Dense(10, activation='softmax')
    ])
    return model


def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def evaluate_model(model, x_test, labels_test):
    outputs = model.predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)
    misclassified = np.sum(labels_predicted != labels_test)
    print('Percentage misclassified =', 100 * misclassified / labels_test.size)


def visualize_predictions(model, x_test, labels_test):
    plt.figure(figsize=(16, 4))
    for i in range(8):
        ax = plt.subplot(2, 8, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray_r')
        plt.title(labels_test[i])
        ax.axis('off')

        output = model.predict(x_test[i].reshape(1, 28, 28, 1))
        plt.subplot(2, 8, i + 9)
        plt.bar(range(10), output[0])
        plt.title(np.argmax(output))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    x_train, y_train, x_test, y_test, labels_test = load_and_preprocess_data()
    learning_rate = 0.0012  # Example of a smaller learning rate

    # Create the optimizer with the desired learning rate
    adam_optimizer = Adam(learning_rate=learning_rate)

    net = build_model()
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

    plot_training_history(history)
    net.save("network_for_mnist.h5")
    evaluate_model(net, x_test, labels_test)
    visualize_predictions(net, x_test, labels_test)







