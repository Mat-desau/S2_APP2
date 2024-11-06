import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Hz = 10000                                                         #changable
Periode = 1 / Hz
dt = Periode / 10000
debut = 0
fin = Periode-dt
difference = fin - debut

CAR = np.arange(debut, fin, dt)
CARRER = 5*signal.square(2*np.pi*CAR*Hz, duty=0.5)+5

W0 = Hz*np.pi                                              #changable
t = np.arange(0, Periode-dt, dt)
num = 6                                                   #changable

A1 = np.zeros((num, int(((Periode)/dt)-dt)), dtype=complex)

for k in range(0, num):
    A1[k,:] = np.exp(-1j*k*W0*t)

A2 = np.zeros(num, dtype=complex)

for k in range(0, num):
    A2[k] = (1/Periode) * np.sum(CARRER * A1[k]) * dt

A3 = np.abs(A2[0]) * np.ones_like(t)

for k in range(1, num):
    A3 += 2 * np.abs(A2[k]) * np.cos(k*W0*t + np.angle(A2[k]))


k = np.pi
Qmax = 12
dq = 4
q = np.arange(dq, Qmax + dq, dq)
df = 1000
f0 = 2000
fmax = 7000
fmin = 1000
f = np.arange(fmin, fmax, df)
fn = f/f0

H = np.zeros((len(f), (len(q))), dtype=complex)

for x in range(0, (len(q))):
    H[:, x] = 20 * np.log(-k/(1 + 1j * q[x] * (fn - 1/fn)))

H2 = np.zeros(len(f), dtype=complex)

H2 = 20 * np.log(-k/(1 + 1j * Qmax * (fn - 1/fn)))

plt.figure()
a = plt.subplot(3, 4, 1)
b = plt.subplot(3, 4, 2)
c = plt.subplot(3, 4, 3)
d2 = plt.subplot(3, 4, 4)
d = plt.subplot(3, 1, 2)
e = plt.subplot(3, 1, 3)

a.plot(CAR, CARRER)
a.grid(color='b', linestyle='-', linewidth=0.1)
a.set_ylabel('Amplitude')
a.set_xlabel('Temps')

b.stem(range(0, num), np.abs(A2))
b.grid(color='b', linestyle='-', linewidth=0.1)
b.set_ylabel('Gain')
b.set_xlabel('Harmonique')

c.plot(t, A3)
c.grid(color='b', linestyle='-', linewidth=0.1)
c.set_ylabel('Amplitude')
c.set_xlabel('Temps')

d.plot(f, np.abs(H))
d.semilogx(f, H)
d.grid(color='b', linestyle='-', linewidth=0.1)
d.set_ylabel('Gain (dB)')
d.set_xlabel('Frequence (Hz)')

e.stem(f, np.abs(H2))
e.grid(color='b', linestyle='-', linewidth=0.1)
e.set_ylabel('Gain (dB)')
e.set_xlabel('Frequence (Hz)')

d2.stem(range(0, num), np.angle(A2))
d2.grid(color='b', linestyle='-', linewidth=0.1)
d2.set_ylabel('Phase')
d2.set_xlabel('Harmonique')

plt.show()
