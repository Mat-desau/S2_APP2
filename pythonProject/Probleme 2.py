import numpy as np
import matplotlib.pyplot as plt

k = 1
Qmax = 40
Qmin = 8
dq = 4
q = np.arange(4, Qmax + dq, dq)
df = 1/1000
f0 = 10
fmax = 2*f0
fmin = 0.5*f0
f = np.arange(fmin, fmax, df)
fn = f/f0

H = np.zeros((len(f), (len(q))), dtype = complex)

for x in range(0, (len(q))):
    H[:, x] = -k/(1 + 1j * q[x] * (fn - 1/fn))

plt.figure()
a = plt.subplot(1, 1, 1)
a.plot(f, np.abs(H))
a.set_ylabel('Gain')
a.set_xlabel('Fn = f / f0')


plt.show()




