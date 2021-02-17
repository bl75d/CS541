import numpy as np
# A = np.array([[1, 2, 3, 5], [11,2, 10, 16], [31, 7, 8, 9], [19, 17, 8, 2]])
A=np.asarray([[1,0,3],[1,2,3],[1,2,3],[1,2,3]])
B=np.asarray([[1,2,3],[1,2,1],[1,2,3],[1,2,3]])

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
C=[[1,0,0,0],[0,0,1,0],[0,0,0,1,],[1,0,0,0]]
print(np.argmax(np.asarray(C),axis=1))