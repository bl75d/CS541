#HW2 xli14@wpi.edu
import numpy as np 

def to_onehot(labels_dense):
	labels_onehot = np.zeros((labels_dense.shape[0],10))
	index_offset = np.arange(labels_dense.shape[0])*10
	labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_onehot

def prediction(X, w, b):
	z = X.dot(w) + b
	exp = np.exp(z)
	temp = np.sum(exp, axis = 1)
	denominator = np.array(temp).reshape(len(temp),-1)
	return exp/denominator

def accuracy(X,y,w,b):
	yhead = prediction(X, w, b)
	label = yhead.argmax(axis = 1)
	label_trans = np.array(label).reshape(len(label),-1)
	classes = to_onehot(label_trans)
	return sum(map(sum,np.abs(classes-y)))/y.shape[0]/2

def cross_entropy(X, y, w, b, alpha):
	return -sum(np.mean(y*np.log(prediction(X,w,b)),axis = 0)) + alpha*np.trace(np.dot(w.T,w))/2

def update_param(X, y, w, b, alpha, lr):
	df_dw = np.mean(np.dot(X.T,prediction(X,w,b)-y), axis = 0) + alpha*w
	df_db = np.mean(prediction(X,w,b)-y, axis = 0)
	w = w - lr*df_dw
	b = b - lr*df_db
	return w, b

def init_param(X):
	w = np.random.rand(X.shape[1],10)
	b = [1 for _ in range(10)]
	return w, b

def train(X, y, alpha, lr, epoch, mini_batch):
	w, b = init_param(X)
	n = int(X.shape[0]/mini_batch)
	for i in range(epoch):
		for j in range(n):
			Xtrain = X[j*mini_batch:(j+1)*mini_batch]
			ytrain = y[j*mini_batch:(j+1)*mini_batch]
			# print('ytrain.shape',ytrain.shape)
			w, b = update_param(Xtrain, ytrain, w, b, alpha, lr)
	return w,b

def L2_SGD_softmax():
	mini_batchs=[10,20,25,50]
	learning_rates=[0.0001,0.0005,0.001,0.005]
	epochs=[5,10,20,50]
	alphas=[0.0001,0.001,0.01,0.1]

	W = [[]]
	B = []
	min_cost = float('inf')
	acc = 0

	for al in alphas:
		for lr in learning_rates:
			for ep in epochs:
				for mb in mini_batchs:
					# print('you are here')
					w, b = train(Xvali,yvali,al,lr,ep,mb)
					ce= cross_entropy(Xvali, yvali, w, b, al)
					if ce < min_cost:
						min_cost = ce
						W, B = w, b
						acc = accuracy(Xvali,yvali,W,B)
						print('ce = ',ce)
						print('accuracy =',acc)
	print('Cross Entropy Loss= '+ str(min_cost))
	print('Accuracy =' + str(acc))
	return W,B,min_cost

if __name__ == '__main__':
	X = np.reshape(np.load('fashion_mnist_train_images.npy'), (-1, 28 * 28)) / 255
	Xte = np.reshape(np.load('fashion_mnist_test_images.npy'), (-1, 28 * 28)) / 255
	y = np.reshape(np.load('fashion_mnist_train_labels.npy'), (-1, 1))
	yte = np.reshape(np.load('fashion_mnist_test_labels.npy'), (-1, 1))

	y, yte = to_onehot(y), to_onehot(yte)
	
	np.random.shuffle(X)
	np.random.shuffle(y)
	nx, ny = int(0.2*X.shape[0]), int(0.2*y.shape[0])
	Xtr, Xvali = X[nx:,:], X[:nx,:]
	ytr, yvali = y[ny:,:], y[:ny,:]

	L2_SGD_softmax()






