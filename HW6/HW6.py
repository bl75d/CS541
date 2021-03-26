import  numpy as np
import keras, os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

def load_data():
    y=np.load("HW6/ages.npy")
    X=np.load("HW6/faces.npy")
    # X = X.reshape(X.shape[0], X.shape[1],X.shape[2], 1)/255
    X=np.repeat(X[:,:,:,np.newaxis],3,axis=3)
    # X = np.reshape(np.load("faces.npy"), (-1, 48 * 48)) / 255
    # y = np.reshape(np.load("ages.npy"), (-1, 1))
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    print(X.shape)
    print(y.shape)
    return X_train,X_test,y_train,y_test

def vgg16():
    model = Sequential()
    model.add(Conv2D(input_shape=(48, 48, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=1, activation="softmax"))

    from keras.optimizers import Adam
    opt = Adam(lr=0.00001)
    model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    return model

if __name__ == "__main__":
    X_train,X_test,y_train,y_test=load_data()
    model=vgg16()
    model.fit(X_train,y_train,
                batch_size=64,
                epochs=20,
                validation_data=(X_test, y_test))
    # Evaluate the model on test set
    score = model.evaluate(X_test, y_test, verbose=0)
    # Print test accuracy
    print('\n', 'Test RMSE:', score[1])