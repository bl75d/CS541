# HW3
# from numba import jit, cuda
import wget
import numpy as np
def download_files():
    urls=["https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy","https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy",
          "https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy","https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy"]
    for url in urls:
        wget.download(url)

def delete_files():
    import os
    files=["fashion_mnist_train_images.npy","fashion_mnist_train_labels.npy",
    "fashion_mnist_test_images.npy","fashion_mnist_test_labels.npy"]
    for f in files:
        os.remove(f)

def z(X,w,b):
    z=X.dot(w) - b
    exp=np.exp(z)/np.sum(np.exp(z),axis=1).reshape(-1,1)
    return exp

# Softmax layer to choose the larget bit as predicting class
def softmax(yhat):
    return np.argmax(np.asarray(yhat),axis=1)

def cost(X,y,w,b,alpha):
    w=np.asarray(w)
    yhat=np.log(z(X,w,b))
    c=[]
    for i in range(X.shape[0]):
        c.append(yhat[i,int(y[i])])
    ce=-np.mean(c)+alpha/2*np.mean(w.dot(w.T))
    return ce

def initial_w_b(X,y):
    w=np.random.rand(X.shape[1],y.shape[1])
    b=1
    return w,b

def update_weights(X,y,w,b,alpha,lr):
    # -x(y - (wx + b))
    DL_dw=X.T.dot(z(X,w,b)-y)
    # -(y - (mx + b))
    DL_db=(z(X,w,b)-y)
    w=(1-alpha*lr)*w-lr*DL_dw
    if np.abs(lr*DL_db.mean())<np.inf:
        b-=lr*DL_db.mean()
    return w,b

def train(data_tr,learning_rate,alpha,mini_batch,epoch):
    data_X=data_tr[:,:-1]
    data_y=data_tr[:,-1].reshape(-1,1)
    data_y=onehotencoding(data_y)
    w,b=initial_w_b(data_X,data_y)
    iter=int(data_X.shape[0]/mini_batch)

    for i in range(epoch):
        for j in range(iter):
            X=data_X[j*mini_batch:(j+1)*mini_batch,:]
            y=data_y[j*mini_batch:(j+1)*mini_batch,:]
            w,b=update_weights(X,y,w,b,alpha,learning_rate)
    return w,b

def split_data():

    X = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28 * 28)) / 255
    y = np.reshape(np.load("fashion_mnist_train_labels.npy"), (-1, 1))
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    n=int(0.2*X.shape[0])
    data_tr,data_vali=data[n:,:],data[:n,:]
    return data_tr,data_vali

def onehotencoding(y):
    from sklearn.preprocessing import OneHotEncoder
    enc_y=OneHotEncoder().fit_transform(y)
    return enc_y

# @cuda.jit
def Softmax_LinearReression():

    # define a global variable as the best penalty in computing Cross Enropy after training
    global alp

    # mini_batchs=[10,20,25,50]
    # learning_rates=[0.0001,0.0005,0.001,0.005]
    # epochs=[5,10,20,50]
    # alphas=[0.0001,0.001,0.01,0.1]

    mini_batchs=[20,50]
    learning_rates=[0.0001,0.0005]
    epochs=[5,10]
    alphas=[0.001,0.01]

    # mini_batchs=[20]
    # learning_rates=[0.0001]
    # epochs=[5]
    # alphas=[0.001]

    min_cost=-1
    W=[]
    B=0

    # Preparing data
    data_tr, data_vali = split_data()
    X_v = data_vali[:, :-1]
    y_v = data_vali[:, -1].reshape(-1, 1)

    # Run training with different preset hyperparameters
    for lr in learning_rates:
        for al in alphas:
            for ep in epochs:
                for mb in mini_batchs:
                    w,b=train(data_tr,learning_rate=lr,alpha=al,mini_batch=mb,epoch=ep)
                    ce=cost(X_v, y_v, w, b,al)
                    if min_cost==-1 or ce<min_cost:
                        min_cost=ce
                        W=w
                        B=b
                        alp=al
    # print("CE:"+str(min_cost))
    return W,B

def evaluation(w,b):
    X_te = np.reshape(np.load('fashion_mnist_test_images.npy'), (-1, 28 * 28)) / 255
    y_te = np.reshape(np.load('fashion_mnist_test_labels.npy'), (-1, 1))
    y_pre=softmax(z(X_te,w,b)).reshape(-1,1)

    # Cross Enropy for validation set
    ce = cost(X_te, y_te, w, b, alp)
    print("Cross Enropy:"+str(ce))

    # Accuracy Score
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_te,y_pre)
    print("Accuracy:"+str(acc))
    return acc,ce


if __name__ == '__main__':
    download_files()
    import time
    start = time.time()
    W,B=Softmax_LinearReression()
    evaluation(W,B)
    print("Running Time:"+str(time.time()-start))
    delete_files()