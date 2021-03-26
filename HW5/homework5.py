#################################################################
# Insert TensorFlow code here to complete the tutorial in part 1.
#################################################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
# import matplotlib.pyplot as plt
def part1_cnn():
    # Load the fashion-mnist pre-shuffled train data and test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
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
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2,padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # # Take a look at the model summary
    # model.summary()
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    model.fit(x_train,
             y_train,
             batch_size=64,
             epochs=10,
             validation_data=(x_test, y_test))

    # Evaluate the model on test set
    score = model.evaluate(x_test, y_test, verbose=0)
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])


#################################################################
# Insert TensorFlow code here to *train* the CNN for part 2.
#################################################################
def part2_cnn():
    # Load the fashion-mnist pre-shuffled train data and test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    x_train = x_train/ 255
    x_test = x_test / 255
    model = tf.keras.Sequential()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(y_train.reshape(-1, 1))
    y_test = enc.fit_transform(y_test.reshape(-1, 1))

    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
    # model.add(layers.Activation(activation='relu'))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # # Take a look at the model summary
    # model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=1,
              validation_data=(x_test, y_test))

    # Evaluate the model on test set
    score = model.evaluate(x_test, y_test, verbose=0)
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])
    model.summary()
    model.trainable_variables
    model.save("cnn_model")
    return model

def part2_prediction(x_train,model):
    yhat1 = model.predict(x_train[0:1,:,:,:])[0]  # Save model's output
    return yhat1
#################################################################
# Write a method to extract the weights from the trained
# TensorFlow model. In particular, be *careful* of the fact that
# TensorFlow packs the convolution kernels as KxKx1xF, where
# K is the width of the filter and F is the number of filters.
#################################################################

def convertWeights (model):
    # Extract W1, b1, W2, b2, W3, b3 from model.
    k=model.trainable_variables

    for i in model.trainable_variables:
        print(i.shape)
    W1, b1, W2, b2, W3, b3=k[0],k[1],k[2],k[3],k[4],k[5]
    return W1, b1, W2, b2, W3, b3

#################################################################
# Below here, use numpy code ONLY (i.e., no TensorFlow) to use the
# extracted weights to replicate the output of the TensorFlow model.
#################################################################

# Implement a fully-connected layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def fullyConnected(W, b, x):
    pass
    return W.T.dot(x) + b

# Implement a max-pooling layer. For simplicity, it only needs
# to work on one example at a time (i.e., does not need to be
# vectorized across multiple examples).
def maxPool (input, poolingWidth,stride):
    pass
    output=[]
    input = np.rollaxis(input, 2)#convert(26,26,64) to (64,26,26)
    for x in input:
        n_rows=int(x.shape[0]/stride)
        n_columns=int(x.shape[1]/stride)
        featuremap=np.zeros((n_rows,n_columns))
        for i in range(n_rows):
            row=[]
            for j in range(n_columns):
                    pool=[]
                    for m in range(poolingWidth):
                        for n in range(poolingWidth):
                            pool.append(x[i*stride+m,j*stride+n])
                    featuremap[i,j]=np.max(pool)
        output.append(featuremap)
    output=np.asarray(output)
    output = np.rollaxis(output, 0, 3)#convert(64,13,13) to(13,13,64)
    # print(output.shape)
    return output

# Implement a softmax function.
def softmax (x):
    pass
    return np.exp(x)/np.sum(np.exp(x))

# Implement a ReLU activation function
def relu (x):
    pass
    x[x<0]=0
    return x

def load_model():
    model=keras.models.load_model("cnn_model")
    return model

# Conv layer
def conv2d(x,W1,b1):
    filterwidth=W1.shape[0]
    num_filters=W1.shape[3]
    output=[]
    for k in range(num_filters):
        featuremap = np.zeros((x.shape[0] - filterwidth + 1, x.shape[1] - filterwidth + 1))
        fltr=W1[:,:,0,k]
        for i in range(x.shape[0]-filterwidth+1):
            for j in range(x.shape[1]-filterwidth+1):

                window = np.zeros((filterwidth, filterwidth))
                for m in range(filterwidth):
                    for n in range(filterwidth):
                        window[m,n]=x[i+m,j+n]
                featuremap[i,j]=np.sum(window*fltr)+b1[k]
        output.append(featuremap)
    output=np.asarray(output)
    output=np.rollaxis(output,0,3)
    return output

# Implement the CNN with the same architecture and weights
# as the TensorFlow-trained model but using only numpy.

# Implement Forward propagation CNN
def NN_forward_propagation(x, W1, b1, W2, b2, W3, b3):
    # conv2d layer
    tensor = conv2d(x, W1, b1)  # (26, 26, 64)

    # maxpool layer
    tensor = maxPool(tensor, 2, 2)

    # relu layer
    tensor = relu(tensor)

    # flatten layer
    tensor = tensor.flatten()

    # fully connected layer1
    tensor = fullyConnected(np.asarray(W2), np.asarray(b2), tensor)
    tensor = relu(tensor)

    # # fully connected layer2
    tensor = fullyConnected(np.asarray(W3), np.asarray(b3), tensor)

    # softmax layer
    yhat = softmax(tensor)
    return yhat

if __name__ == '__main__':

    # Run the function to get weights and biases files
    # model=part2_cnn()

    # load the model to get the weights and biases if 'cnn_model' file exists
    model=load_model()
    print(model.summary())
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Load weights from TensorFlow-trained model.
    W1, b1, W2, b2, W3, b3 = convertWeights(model)
    # print(W1[:,:,0,0].shape)
    x=x_test[0, :, :, :] #input is a 28*28*1 image

    yhat=NN_forward_propagation(x, W1, b1, W2, b2, W3, b3)
    # yhat1=np.argmax(yhat1)


    # tf model prediction
    yhat2=part2_prediction(x_test,model)
    print("TF CNN prediction:")
    print(yhat2)
    
    # fully connected model prediction
    print("Fully connected NN prediciton:")
    print(yhat)
