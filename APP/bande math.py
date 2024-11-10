import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

k = 1
Qmax = 20
Qmin = 1
dq = 1
q = np.arange(Qmin, Qmax + dq, dq)
df = 1/100
f0 = 5000
fmax = 9000
fmin = 1000
f = np.arange(fmin, fmax, df)

H = np.zeros((len(f), (len(q))), dtype = complex)

for x in range(0, (len(q))):
    H[:, x] = -k/(1 + 1j * q[x] * (f/f0 - f0/f))

Hz = 1000                                                         #changable
Periode = 1 / Hz
dt = Periode / 1000
debut = 0
fin = Periode-dt
difference = fin - debut

CAR = np.arange(debut, fin, dt)
CARRER = 5*signal.square(2*np.pi*CAR*Hz, duty=0.5)+5

W0 = Hz*2*np.pi                                              #changable
t = np.arange(0, Periode-dt, dt)
num = 6                                                   #changable

A1 = np.zeros((num, int(Periode/dt)), dtype=complex)

for k in range(0, num):
    A1[k,:] = np.exp(-1j*k*W0*t)

A2 = np.zeros(num, dtype=complex)

for k in range(0, num):
    A2[k] = (1/Periode) * np.sum(CARRER * A1[k]) * dt

plt.figure()
a = plt.subplot(3, 1, 1)
b = plt.subplot(3, 1, 2)
c = plt.subplot(3, 1, 3)

a.plot(CAR, CARRER)
a.grid(color='b', linestyle='-', linewidth=0.1)
a.set_ylabel('Amplitude')
a.set_xlabel('Temps')

b.stem(range(0, num), np.abs(A2))
b.grid(color='b', linestyle='-', linewidth=0.1)
b.set_ylabel('Gain')
b.set_xlabel('Harmonique')

c.stem(range(0, num), np.angle(A2))
c.grid(color='b', linestyle='-', linewidth=0.1)
c.set_ylabel('Amplitude')
c.set_xlabel('Temps')

plt.figure()
a = plt.subplot(1, 1, 1)
a.plot(f, np.abs(H[:,1-1]))
a.plot(f, np.abs(H[:,5-1]))
a.plot(f, np.abs(H[:,10-1]))
a.plot(f, np.abs(H[:,12-1]))
a.plot(f, np.abs(H[:,14-1]))
a.set_ylabel('Gain')
a.set_xlabel('frequence')


plt.show()