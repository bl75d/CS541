import numpy as np
# A = np.array([[1, 2, 3, 5], [11,2, 10, 16], [31, 7, 8, 9], [19, 17, 8, 2]])
# A=np.asarray([[1,0,3],[1,2,3],[1,2,3],[1,2,3]])
# B=np.asarray([[1,2,3],[1,2,1],[1,2,3],[1,2,3]])

# def problem_1k (A, k):
#     A=np.array(A)
#     vals,vecs=np.linalg.eig(A)
#     print(vals)
#     print("*****")
#
#     print(vecs)
#     print("*****")
#
#     idx = vals.argsort()[-k:][::-1]
#     print(idx)
#     print("*****")
#
#     v=vecs[idx]
#     # print(v)
#     return vecs[:,idx]
#     # return vecs
# print(problem_1k(A,3))
# print(np.sum(A*B,axis=1).mean())
# C=[[1,0,0,0],[0,0,1,0],[0,0,0,1,],[1,0,0,0]]
# print(np.argmax(np.asarray(C),axis=1))

# B=np.asarray([[1,2,3],[1,0,1],[1,2,3],[1,2,3]])
# d=np.asarray([[1,1,1]])
# print(B+d)
# x=np.asarray([[1,2,3],[6,6,6],[2,2,2],[1,1,1]])
# avg=np.sum(x,axis=1).reshape(-1,1)
#
# div=x/np.sum(x,axis=1).reshape(-1,1)
# print(div.shape)
# print(div)
# print(np.sum(x,axis=1))
# y=np.asarray([[1],[1],[3]])
# B=np.asarray([[3,3,2]],[4,4,4])
# print(np.stack([A.flatten() for a in A]+[B.flatten() for b in B]))
# W = 2 * (np.random.random(size=(10, 5)) / 5 ** 0.5) - 1. / 5 ** 0.5
# print(W.shape)
# print(W)
# a=[[2,3,1],[1,2,3]]

# c=np.stack(a,y)
# print(c)
x=np.asarray([[1,2,3],[6,6,6],[2,2,2],[1,1,1]])
y=np.asarray([[1,2,3],[6,6,6],[2,2,2],[1,1,1]])
# c=x.flatten()
# d=y.flatten()
# c=np.vstack((c,d))
# print(np.vstack((c,d)))
# c=[]
# c.append(x.flatten())
# c.append(x.flatten())
# c.append(x.flatten())
# c=np.asarray(c)
# print(c)

# a=np.asarray([-3.82232105, -6.25759225, -4.9874595,  -5.78260876, -3.91829208,8.01789327])
# def softmax (x):
#     pass
#     return np.exp(x)/np.sum(np.exp(x))
# print(np.argmax(softmax(a)))
import tensorflow as tf 
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)