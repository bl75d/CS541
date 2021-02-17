import numpy as np

# B=np.asarray([[1,2,3],[1,2,3],[1,2,3]])
A=np.asarray([[1,1,1]])
B=np.asarray([[2,2,2]])
C=np.asarray([[1,1,1]])
# A = [[1, 4, 5],
#     [-5, 8, 9]]
# B = [[1, 4, 5],
#     [-5, 8, 9]]
# print(A.shape)
# print(B.shape)
# print(C.shape)

A = np.array([[1, 2], [3, 5]])
x = np.array([1, 2])
# y=np.array([[1],[2]])

def problem_1a (A, B):
    return A + B
# print(problem_1a(A, B))

def problem_1b (A, B, C):
    A=np.asarray(A)
    B=np.asarray(B)
    C=np.asarray(C)
    return A.dot(B)-C
# print(problem_1b(A, B, C))

def problem_1c (A, B, C):
    A=np.asarray(A)
    B=np.asarray(B)
    C=np.asarray(C)
    return A*B+C.T
# print(problem_1c(A, B, C))

def problem_1d (x, y):
    return np.inner(x,y)
# print(problem_1d(A, B))

def problem_1e (A):
    A=np.asarray(A)
    return np.zeros(A.shape)
# print(problem_1e(A))

def problem_1f (A, x):
    A=np.array(A)
    x=np.array(x)
    return np.linalg.solve(A,x)
# print(problem_1f(A,x))

def problem_1g (A, x):
    A=np.array(A)
    x=np.array(x)
    return np.linalg.solve(A.T,x.T).T
# print(problem_1g(A,x))

# A = [[1, 4],
#     [-5, 9]]
# alpha=2
def problem_1h (A, alpha):
    A=np.array(A)
    I=np.eye(A.shape[0])
    print(I)
    return A+alpha*I
# print(problem_1h(A,alpha))

# A=[[3, 1, 7, 2, 9, 3, 1, 4],
#     [3, 6, 7, 8, 9, 2, 1, 5]]
# i=0
def problem_1i (A, i):
    A=np.array(A)
    return np.sum(A[i,1::2])
# print(problem_1i(A,i))

# A = np.array([[1, 2, 3], [11, 10, 16], [7, 8, 9]])
def problem_1j (A, c, d):
    A=np.array(A)
    out=A[np.nonzero(np.logical_and(A>=c,A<=d))]
    return np.mean(out)
# print(problem_1j(A,2,8))

# A = np.array([[1, 2, 3, 5], [11,2, 10, 16], [31, 7, 8, 9], [19, 17, 8, 2]])
def problem_1k (A, k):
    A=np.array(A)
    vals,vecs=np.linalg.eig(A)
    idx = vals.argsort()[-k:][::-1]
    return vecs[:,idx]
# print(problem_1k(A,2))

# x=np.array([1,1,1,1])
def problem_1l (x, k, m, s):
    x=np.array(x)
    n=x.shape[0]
    mean=np.ones(n)
    cov=np.identity(n)
    out=np.random.multivariate_normal(m*mean,s*cov,k)
    print(out.T.shape)
    return out.T
# print(problem_1l(x,5,1,0.5))

# A = np.array([[1, 2, 3, 5], [11,2, 10, 16], [31, 7, 8, 9], [19, 17, 8, 2]])
def problem_1m (A):
    A=np.array(A)
    rng = np.random.default_rng()
    return rng.permutation(A,axis=0)
# print(problem_1m(A))
# x=[3, 1, 7, 2, 9, 3, 1, 4]
def problem_1n (x):
    x=np.array(x)
    return (x-np.mean(x))/np.std(x)
# print(problem_1n(x))

# x=[3, 1, 7, 2, 9, 3, 1, 4]
def problem_1o (x, k):
    return np.repeat(np.atleast_2d(x),k,axis=0)
# print(problem_1o(x,3))

# X=[[1,2,3,4], [1,2,3,4],[1,2,3,4]]
def problem_1p (X):
    D=np.atleast_3d(X)
    d3=np.repeat(D,D.shape[1],axis=2)
    nd1=np.swapaxes(d3,0,2)
    # print(nd1)
    nd2=np.swapaxes(nd1,0,1)
    # print(nd2)
    sub=(nd2-nd1)**2
    return np.sum(sub,axis=2)
# print(problem_1p(X))

def linear_regression (X_tr, y_tr):
    A=X_tr.T.dot(X_tr)
    x=X_tr.T.dot(y_tr)
    return np.linalg.solve(A,x)




def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("HW2/age_regression_Xtr.npy"), (-1, 48 * 48))
    ytr = np.load("HW2/age_regression_ytr.npy")
    X_te = np.reshape(np.load("HW2/age_regression_Xte.npy"), (-1, 48 * 48))
    yte = np.load("HW2/age_regression_yte.npy")
    w = linear_regression(X_tr, ytr)
    print(w)

    # Report fMSE cost on the training and testing data (separately)
    mse_tr=(np.square(X_tr.dot(w) - ytr)).mean()/2
    mse_te=(np.square(X_te.dot(w) - yte)).mean()/2
    return mse_tr,mse_te
# print(train_age_regressor())



# def HW2_2():
#     from sklearn.neural_network import MLPRegressor
#     from sklearn.model_selection import train_test_split
#     X = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
#     y = np.load("age_regression_ytr.npy")
#     X_tr,X_te,ytr,yte=train_test_split(X,y,test_size=0.2)
#
#     alf=0.0001
#     n=200
#     lr='adaptive'
#     epoch=500
#     reg = MLPRegressor(hidden_layer_sizes=2,solver='sgd',alpha=alf,batch_size=n,learning_rate=lr,max_iter=epoch).fit(X_tr, ytr)
#     print(reg.score(X_te,yte))
#
#     return reg
# print(HW2_2())


# # HW2
# def initial_w_b():
#     X = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
#     w=np.random.rand(X.shape[1],1)
#     b=np.random.rand(X.shape[1],1)
#     return w,b
#
# def L2_SGD_Reg_LinearReression():
#     mini_batch=[20,50,100,500]
#     learning_rate=[0.001,0.005,0.01,0.1]
#     epoch=[100,200,500,1000]
#     L2=[0.001,0.01,0.1,0.5]
#
#     w,b=initial_w_b()





# HW2
def prediction(X,w,b):
    y=X.dot(w)-b
    # print(y.shape)
    return X.dot(w)-b

def cost(X,y,w,b):
    return (np.square(prediction(X,w,b) - y)).mean()/2

def update_weights(X,y,w,b,alpha,mini_batch,lr):
    DL_dw=0
    DL_db=0

    # -2x(y - (wx + b))
    y=y.reshape(-1,1)
    # print(y.shape)
    # print(prediction(X,w,b).shape)
    # print(X.shape)
    DL_dw=X.T.dot(prediction(X,w,b)-y)
    # -2(y - (mx + b))
    DL_db=(prediction(X,w,b)-y)

    w-=(1-alpha*lr)*w-lr*DL_dw
    b-=lr*DL_db
    return w,b


def initial_w_b(X):
    # X = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    w=np.random.rand(X.shape[1],1)
    # b=np.random.rand(X.shape[0],1)
    b=np.random.normal()
    # print(w.shape)
    return w,b

def train(X,y,lr,alpha,mini_batch,epoch):
    w,b=initial_w_b(X)
    # iter=int(X.shape[0]/mini_batch)
    #
    # for i in range(epoch):
    #     for j in range(iter):
    #         w,b=update_weights(X,y,w,b,alpha,mini_batch,lr)
    #         cost=cost(X,y,w,b)
    #         prediction(cost)
    w, b = update_weights(X, y, w, b, alpha, mini_batch, lr)
    print(cost(X,y,w,b))
    return cost(X,y,w,b)


def L2_SGD_Reg_LinearReression():
    mini_batch=[20,50,100,500]
    learning_rate=[0.001,0.005,0.01,0.1]
    epoch=[100,200,500,1000]
    alpha=[0.001,0.01,0.1,0.5]

    X = np.reshape(np.load("HW2/age_regression_Xtr.npy"), (-1, 48 * 48))
    y = np.load("HW2/age_regression_ytr.npy")
    train(X,y,lr=0.01,alpha=0.001,mini_batch=10,epoch=10)


print(L2_SGD_Reg_LinearReression())


































































