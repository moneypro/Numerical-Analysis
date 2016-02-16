import numpy as np
def pivot(m):
    # n = len(A)
    # P = np.eye(n)
    # for j in xrange(n):
    #     i = max(xrange(j, n), key=lambda k: abs(A[k][j]))
    #     if j != i:
    #         P[j], P[i] = P[i], P[j]
    # return P
    #     """Creates the pivoting matrix for m."""
    n = len(m)
    ID = [[float(i == j) for i in xrange(n)] for j in xrange(n)]
    # print (ID)
    # print (np.eye(n))
    for j in xrange(n):
        row = max(xrange(j, n), key=lambda i: abs(m[i][j]))
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return np.array(ID)

def lu(A):
    """Decomposes a nxn matrix A by PA=LU and returns L, U and P."""
    n = len(A)
    L = [[0.0] * n for i in xrange(n)]
    U = [[0.0] * n for i in xrange(n)]
    P=0
    # P = pivotize(A)
    # A2 = matrixMul(P, A)
    for j in xrange(n):
        L[j][j] = 1.0
        for i in xrange(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in xrange(i))
            U[i][j] = A[i][j] - s1
        for i in xrange(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in xrange(j))
            L[i][j] = (A[i][j] - s2) / U[j][j]
    return (L, U, P)

A=np.array([[1,0,1],[2,1,0],[3,4,5]])
b=np.array([1,0,2]).transpose()
P=pivot(A)
print (P)
U=np.zeros(A.shape)
L=np.eye(A.shape[0])
for i in range(len(A)):
    U[i]=A[i]
    for j in range(0,i):
        L[i,j] = U[i,j]/U[j,j]
        U[i]-=U[i,j]/U[j,j]*U[j]
# print (A)

x=np.zeros(A.shape[0])
y=np.zeros(A.shape[0])
for i in range(0,len(A)):
    acc = b[i]
    for j in range(0,i):
        acc -=L[i,j]*y[j]
    y[i]=acc/L[i,i]
# print (y)
# L_inv = 2*np.eye(A.shape[0])-L
# print (L_inv)
# Linvb = np.dot(L_inv , b)
# print (Linvb)
# x[A.shape[0]-1]= Linvb[A.shape[0]-1]
for i in range(len(A)-1,-1,-1):
    acc = y[i]
    for j in range(i+1,len(A)):
        acc -=U[i,j]*x[j]
    x[i]=acc/U[i,i]
# print (x)
# print (np.linalg.solve(A,b))

# L2,U2,P2=lu(A)
# print(L2)
# print(U2)
# print (U)
# print (L)

