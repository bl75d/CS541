# HW2
import numpy as np
def prediction(X,w,b):
    # print(X.shape)
    # print(w.shape)
    y=X.dot(w)-b
    # print(y.shape)
    return X.dot(w)-b

def cost(X,y,w,b):
    return (np.square(X.dot(w)-b - y)).mean()/2

def update_weights(X,y,w,b,alpha,lr):

    # -x(y - (wx + b))
    DL_dw=X.T.dot(prediction(X,w,b)-y)
    # -(y - (mx + b))
    DL_db=(prediction(X,w,b)-y)
    w=(1-alpha*lr)*w-lr*DL_dw
    if np.abs(lr*DL_db.mean())<np.inf:
        b-=lr*DL_db.mean()
    return w,b


def initial_w_b(X):
    # X = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    w=np.random.rand(X.shape[1],1)
    # b=np.random.rand(X.shape[0],1)
    b=1
    # print(w.shape)
    return w,b

def train(data_tr,learning_rate,alpha,mini_batch,epoch):
    data_X=data_tr[:,:-1]
    data_y=data_tr[:,-1].reshape(-1,1)
    w,b=initial_w_b(data_X)
    iter=int(data_X.shape[0]/mini_batch)

    for i in range(epoch):
        for j in range(iter):
            X=data_X[j*mini_batch:(j+1)*mini_batch,:]
            y=data_y[j*mini_batch:(j+1)*mini_batch,:]
            w,b=update_weights(X,y,w,b,alpha,learning_rate)
            # print(b)
            # cost=cost(X,y,w,b)
            # prediction(cost)
    # w, b = update_weights(X, y, w, b, alpha, learning_rate)
    # print(cost(X,y,w,b))
    return w,b

def train_validation_data():
    X = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    y = np.reshape(np.load("age_regression_ytr.npy"), (-1, 1))
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    n=int(0.2*X.shape[0])
    data_tr,data_vali=data[n:,:],data[:n,:]
    return data_tr,data_vali



def L2_SGD_Reg_LinearReression():
    mini_batchs=[10,20,25,50]
    learning_rates=[0.0001,0.0005,0.001,0.005]
    epochs=[5,10,20,50]
    alphas=[0.0001,0.001,0.01,0.1]
    # mini_batchs=[20,50]
    # learning_rates=[0.0001,0.0005]
    # epochs=[5,10]
    # alphas=[0.001,0.01]

    min_cost=-1
    W=[]
    B=0

    # Preparing data
    data_tr, data_vali = train_validation_data()
    X_v = data_vali[:, :-1]
    y_v = data_vali[:, -1].reshape(-1, 1)
    # print(X_v.shape)
    # print(y_v.shape)
    # Run training with different preset hyperparameters
    for lr in learning_rates:
        for al in alphas:
            for ep in epochs:
                for mb in mini_batchs:
                    w,b=train(data_tr,learning_rate=lr,alpha=al,mini_batch=mb,epoch=ep)
                    mse=cost(X_v, y_v, w, b)

                    if min_cost==-1 or mse<min_cost:
                        min_cost=mse
                        W=w
                        B=b
    print("MSE:"+str(min_cost))
    return w,b,min_cost


if __name__ == '__main__':
    L2_SGD_Reg_LinearReression()