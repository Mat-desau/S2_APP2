import numpy as np
import matplotlib.pyplot as plt

k = np.pi
fDebut = 0.1
fFin = 16
df = 0.1
f = np.arange(fDebut, fFin, df)
w = 2*k * f
val = 1.111



H = np.zeros((len(f)), dtype=complex)

H = ((1j * w)/(1 + 1j * w * val))

H2 = np.zeros((len(f)))

H2 = 20 * np.log10(np.abs(H))

fH3 = (1, 15)

H3 = (H2[int(1/df-1)], H2[int(15/df-1)])

plt.figure()
c = plt.subplot(1, 1, 1)

c.plot(f, np.abs(H))
c.semilogx(f, np.abs(H))
c.grid(color='b', linestyle='-', linewidth=0.1)
c.set_ylabel('Gain')
c.set_xlabel('Frequence (Hz)')

plt.figure()
d = plt.subplot(2, 1, 1)
e = plt.subplot(2, 2, 3)
c = plt.subplot(2, 2, 4)

d.plot(f, H2)
d.semilogx(f, H2)
d.grid(color='b', linestyle='-', linewidth=0.1)
d.set_ylabel('Gain (dB)')
d.set_xlabel('Frequence (Hz)')

e.stem(fH3[0], H3[0])
e.grid(color='b', linestyle='-', linewidth=0.1)
e.set_ylabel('Gain (dB)')
e.set_xlabel('Frequence (Hz)')

c.stem(fH3[1], H3[1])
c.grid(color='b', linestyle='-', linewidth=0.1)
c.set_ylabel('Gain (dB)')
c.set_xlabel('Frequence (Hz)')

plt.show()