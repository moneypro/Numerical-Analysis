import numpy as np
import matplotlib.pyplot as plt

# def not_allowed(*args, **kwargs):
#     raise RuntimeError("You called an illegal function.")

# import scipy.linalg as sla
# for attr in dir(sla):
#     setattr(sla, attr, not_allowed)
# import numpy.linalg as la
# la.solve = not_allowed

# Number of points in a side
n = 30

# Capacitor plate positions
plate_xpos = [9,19]
plate_ymin = 9
plate_ymax = 19

def get_A():
    A = np.zeros((n**2, n**2))
    for i in range(n**2):
        x = i % n
        y = i // n
        if y >= plate_ymin and y <= plate_ymax and x in plate_xpos:
            A[i,i] = 4
            continue
        A[i,i] = -4
        if x < n - 1: A[i,i+1] = 1
        if x > 0:     A[i,i-1] = 1
        if y < n - 1: A[i,i+n] = 1
        if y > 0:     A[i,i-n] = 1
    return A

def get_b():
    sp = np.linspace(0, 1, n)
    xs, ys = np.meshgrid(sp, sp)
    ys_in_range = (ys >= sp[plate_ymin]) & (ys <= sp[plate_ymax])
    ones = np.ones(xs.shape)
    zeros = np.zeros(xs.shape)
    b =  np.where((xs == sp[plate_xpos[0]]) & ys_in_range, -ones, zeros)
    b += np.where((xs == sp[plate_xpos[1]]) & ys_in_range, ones, zeros)
    return 4 * b.ravel()

A = get_A()
b = get_b()
k = n

def plot_solution(soln):
    plt.figure(figsize=(4,4))
    plt.imshow(soln.reshape((n,n)))
    plt.colorbar()

U=np.zeros(A.shape)
L=np.eye(A.shape[0])
for i in range(len(A)):
    U[i]=A[i]
    for j in range(max(i-k,0),i):
        L[i,j] = U[i,j]/U[j,j]
        U[i]-=U[i,j]/U[j,j]*U[j]

x=np.zeros(A.shape[0])
y=np.zeros(A.shape[0])

for i in range(len(A)):
    acc = b[i]
    for j in range(max(i-k,0),i):
        acc -=L[i,j]*y[j]
    y[i]=acc/L[i,i]

for i in range(len(A)-1,-1,-1):
    acc = y[i]
    for j in range(i+1,min(i+1+k,len(A))):
        acc -=U[i,j]*x[j]
    x[i]=acc/U[i,i]

print (x)