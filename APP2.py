import numpy as np
import matplotlib.pyplot as plt

T = 1               #nombre de periodes
A = 1               #amplitude
W0 = np.pi/2        #decalage
dt = 1/1000         #nombre de points
nb = 2             #nombre de reproductions
R = 1000
C = 10*pow(10, -9)
val = 1.111

t = np.arange(0, T, dt)
t2 = np.arange(0, T*nb, dt)
X1 = np.zeros((int(T/dt)), dtype=complex)

w = (1/T)*2*np.pi

for k in range(0, int(T/dt)):
    X1[k] = (A/2) * ((np.exp(1j*w*k*dt)*(np.exp(1j*W0))) + (np.exp(-1j*w*k*dt)*np.exp(-1j*W0)))

X2 = np.zeros_like(X1, dtype=complex)

for k in range(0, int(T/dt)):
    if X1[k] < 0:
        X2[k] = np.abs(X1[k])
    else:
        X2[k] = X1[k]


X3 = np.zeros_like(t2, dtype=complex)

for k in range(0, nb):
    for x in range(0, int(T/dt)):
        z = 1000*k
        X3[z+x] = X2[x]

X4 = np.zeros_like(t2, dtype=complex)
u = np.zeros_like(t2, dtype=complex)
u2 = np.zeros((int((T/dt)*nb), int((T*nb)/dt)), dtype=complex)

a = 30*np.pi              #valeur dans le sin

for e in range(0, int((T/dt)*nb)):
    u[e] = X3[e] - e

for e in range(0, int((T/dt)*nb)):
    u2[e, :] = (np.exp(1j*a*u[e]*t2)) * (np.exp(-1j*a*u[e]*t2)) * (np.exp(((-e)/val))*t2)

for e in range(0, int((T/dt)*nb)):
    X4[e] += (1/val) * np.sum(X3 * u2[e]) * dt


a = plt.subplot(2, 3, 1)
b = plt.subplot(2, 3, 2)
c = plt.subplot(2, 3, 3)
d = plt.subplot(2, 1, 2)
e = plt.subplot(2, 1, 2)
a.plot(t, X1)
b.plot(t, X2)
c.plot(t2, X3)
e.plot(t2, X3)
d.plot(t2, X4)

plt.show()
