import numpy as np
import matplotlib.pyplot as plt

T = 1/15              #nombres_de_periodes
A = 1               #amplitude
W0 = np.pi/2        #decalage
dt = T/1000         #nombre de points
nb = 10             #nombre de reproductions
R = 100000
C = 1*pow(10, -6)
val = R * C
W = 2*np.pi/T

t = np.arange(0, T, dt)
t2 = np.arange(0, T*nb, dt)
t3 = np.arange(0, T*nb*2-dt, dt)
X1 = np.zeros((int(T/dt)), dtype=complex)

w = (1/T)*2*np.pi

X1 = np.abs(A * np.sin(w*t))

X2 = np.zeros_like(t2)

for k in range(0, nb):
    for x in range(0, int(T/dt)):
        z = 1000*k
        X2[z+x] = X1[x]

H = 1/val * np.exp(-t2/val)

X3 = np.convolve(X2, H) * dt

a = plt.subplot(2, 3, 1)
b = plt.subplot(2, 3, 2)
c = plt.subplot(2, 3, 3)
d = plt.subplot(2, 1, 2)
a.plot(t, X1)
b.plot(t2, X2)
c.plot(t2, H)
d.plot(t2, X2)
d.plot(t3, X3)

plt.show()