import numpy as np
import matplotlib.pyplot as plt

T = 4
W0 = (2*np.pi)/T
dt = 1/1000
t = np.arange(0, T, dt)

X1 = np.hstack((np.ones(int(1/dt)), -1*np.ones(int(2/dt)), np.ones(int(1/dt)))) #creation de la ligne avec les tailles ecrite

u = np.zeros((21, int(T/dt)), dtype=complex) # remplir ligne de 0

for k in range(0, 21):
    u[k,:] = np.exp(-1j*k*W0*t) # faire les variables de l'exposant dans la variable u

X = np.zeros(21, dtype=complex)

for k in range(0, 21):
    X[k] = (1/T) * np.sum(X1 * u[k]) * dt #mettre les variable u dans l'integral en utilisant le tableau X1

X2 = np.zeros_like(t)

for k in range(0, 21):
    X2 += 2 * np.abs(X[k]) * np.cos(k*W0*t + np.angle(X[k])) # transferer le tableau dans le nouveau

plt.figure()
a = plt.subplot(2, 2, 1) #cree une page avec 2 espaces et mettre dans la premiere case
a.plot(t, X1) #dessiner le graphique
a.set_xlabel('t')
a.set_ylabel('x(t)')

b = plt.subplot(2, 2, 2)

b.stem(range(0, 21), np.abs(X)) # faire le graphique de barres
b.set_xlabel('k')
b.set_xticks(range(0, 21))
b.set_ylabel('amplitude')


c = plt.subplot(2, 2, 3)

c.stem(range(0, 21), np.angle(X))
c.set_xlabel('k')
c.set_xticks(range(0, 21))
c.set_ylabel('phase')

d = plt.subplot(2, 2, 4)

d.plot(t, X2)
d.set_xlabel('t')

#plt.show() #montrer les graphiques





