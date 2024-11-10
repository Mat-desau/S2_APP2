import numpy as np
import matplotlib.pyplot as plt

T = 1              #nombres_de_periodes
A = 1               #amplitude
W0 = np.pi/2        #decalage
dt = 1/1000         #nombre de points
dtaux = 1/1000      #Quelque chose que Mat voulait
nb = 3             #nombre de reproductions
R = 1000
C = 10*pow(10, -9)
val = 1
W = 2*np.pi/T
Ttaux = np.arange(0, T, dtaux)

t = np.arange(0, T, dt)
t2 = np.arange(0, T*nb, dt)
t3 = np.arange(0, int(T/(dt*4)), dt)
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

X4 = np.zeros((int(T/(dt*4))))

for i in range(0, int(T/(dt*4))):
    X4 = ((np.exp(-i/val)*val*W)/((pow(val, 2) * pow(W, 2))+ 1)) - (((val*W*np.cos(W*i))-np.sin(W*t))/((pow(val, 2)*pow(W, 2))+1))


#X4 = np.zeros(int(T/dt), dtype=complex)
#H = np.zeros(int(T/dt))

#for i in range(0, int(T/dt)):
    #    H[i] = (1/val) * np.exp(-i/val)

#for i in range(0, int(T/dt)):
 #   for taux in range(0, i):
   #     X4[i] = np.sum(H[i] * X2[i - taux]) * dtaux

#u = np.zeros_like(t2, dtype=complex)
#u2 = np.zeros((int((T/dt)*nb), int((T*nb)/dt)), dtype=complex)

            #valeur dans le sin

#for e in range(0, int((T/dt)*nb)):
    #u[e] = X3[e] - e

#for e in range(0, int((T/dt)*nb)):
    #u2[e, :] = (np.exp(1j*a*u[e]*t2)) * (np.exp(-1j*a*u[e]*t2)) * (np.exp(((-e)/val))*t2)

#for e in range(0, int((T/dt)*nb)):
    #X4[e] += (1/val) * np.sum(X3 * u2[e]) * dt



a = plt.subplot(2, 3, 1)
b = plt.subplot(2, 3, 2)
c = plt.subplot(2, 3, 3)
d = plt.subplot(2, 1, 2)
e = plt.subplot(2, 1, 2)
a.plot(t, X1)
b.plot(t, X2)
c.plot(t2, X3)
e.plot(t2, X3)
d.plot(t, X4)

plt.show()
