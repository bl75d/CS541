import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
def cnn():
    # Load the fashion-mnist pre-shuffled train data and test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    x_train = x_train / 255
    x_test = x_test / 255
    model = tf.keras.Sequential()
    x_train=x_train.reshape(x_train.shape[0],28,28,1)
    x_test=x_test.reshape(x_test.shape[0],28,28,1)

    from sklearn.preprocessing import OneHotEncoder
    enc=OneHotEncoder(sparse=False)
    y_train=enc.fit_transform(y_train.reshape(-1,1))
    y_test=enc.fit_transform(y_test.reshape(-1,1))
    # print(x_train.shape)
    # print(y_train.shape)
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    model.fit(x_train,
             y_train,
             batch_size=64,
             epochs=2,
             validation_data=(x_test, y_test))
    # import pickle
    # # save the model to disk
    # filename = 'CNN_Model.sav'
    # with open(filename,'rb') as file:
    #     pickle.dump(model,file)
    model.save("cnn_model")
    # Evaluate the model on test set
    score = model.evaluate(x_test, y_test, verbose=0)
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])
    model.summary()
    # print(model.trainable_variables)
    return model

def load_model():
    model=keras.models.load_model("cnn_model")
    return model


if __name__ == '__main__':
    cnn()
    model=load_model()
    print(model.summary())
    # for i in range(len(model.trainable_variables)):
    #     print(model.trainable_variables[i].shape)