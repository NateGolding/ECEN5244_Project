import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import noise as ns


def dB2lin(db):
    return 10**(db/10)
def lin2dB(db):
    return 10*np.log10(db)

## ULA configuration:
# 3 signals -> 0, -15, 20 degrees
# 8 sensors -> lambda/4 separation
Nsamples = 1024
M = 16
N = 3
theta = np.deg2rad([-28, 36, 3])
d = (1/4)
SNR_dB = 0
cr_min = 0.1
prom_threshold = 0.01        # pct full scale prominence threshold for Pmusic peak finding

# generate array manifold
A = np.exp(-1j*2*np.pi*d*np.sin(theta))**(np.arange(M).reshape(-1,1))

# generate signals, noise, noiseless/noisy sensor readings
SNR = dB2lin(SNR_dB)
npwr = 1/SNR
Rnn = sc.linalg.toeplitz(np.linspace(npwr, 0.75, M))
s = (1/np.sqrt(2))*np.random.randn(N,Nsamples) + (1j/np.sqrt(2))*np.random.randn(N,Nsamples)
n = ns.structured_noise(Rnn, Nsamples)
x = A@s
y = A@s + n

# covariance matrix
Rss = (1/Nsamples) * s@(s.conj().T)
Rnn = (1/Nsamples) * n@(n.conj().T)
Rxx = (1/Nsamples) * x@(x.conj().T)
Ryy = (1/Nsamples) * y@(y.conj().T)

fig,ax = plt.subplots(2,2)
sns.heatmap(np.abs(Rss), ax=ax[0,0], annot=True)
ax[0,0].set_title("Signal Covariance Rss")
sns.heatmap(np.abs(Rnn), ax=ax[0,1], annot=True)
ax[0,1].set_title("Noise Covariance Rnn")
sns.heatmap(np.abs(Rxx), ax=ax[1,0], annot=True)
ax[1,0].set_title("Noiseless Array Covariance Rxx")
sns.heatmap(np.abs(Ryy), ax=ax[1,1], annot=True)
ax[1,1].set_title("Array Covariance Ryy")
plt.show()

# apply covariance differencing
J = np.eye(M,M)[::-1, :]
Pyy = Ryy - J@Ryy@J

fig, ax = plt.subplots(1,2)
sns.heatmap(np.abs(Ryy), ax=ax[0])
ax[0].set_title("Array Covariance")
sns.heatmap(np.abs(Pyy), ax=ax[1])
ax[1].set_title("Differenced Array Covariance")
plt.show()

# eigendcomposition
Dx,Ux = np.linalg.eig(Rxx)
Dy,Uy = np.linalg.eig(Ryy)
Dp,Up = np.linalg.eig(Pyy)

fig,ax = plt.subplots(1,3)
sns.heatmap(np.abs(np.diag(Dx)), ax=ax[0], annot=True)
ax[0].set_title("Noiseless Eigenvalues")
sns.heatmap(np.diag(np.abs(Dy)), ax=ax[1], annot=True)
ax[1].set_title("Noisy Eigenvalues")
sns.heatmap(np.diag(np.abs(Dp)), ax=ax[2], annot=True)
ax[2].set_title("Transformed Eigenvalues")
plt.show()

# psuedospectrum
Uny = Uy[:,(M-N):]
Unp = Up[:,(M-N):]
th = np.linspace(-np.pi/2, np.pi/2, 2048)
Py = np.empty(2048)
Pp = np.empty(2048)
for i in range(2048):
    a_th = np.exp(-1j*2*np.pi*d*np.sin(th[i]))**(np.arange(M).reshape(-1,1))
    Py[i] = 1/np.abs((a_th.conj().T)@Uny@(Uny.conj().T)@a_th).item()
    Pp[i] = 1/np.abs((a_th.conj().T)@Unp@(Unp.conj().T)@a_th).item()

# spectrum peaks
th = np.rad2deg(th)
y_peaks, _ = ss.find_peaks(Py, prominence=(prom_threshold*abs(np.max(Py)-np.min(Py))))
y_arrivals = th[y_peaks]
p_peaks, _ = ss.find_peaks(Pp, prominence=(prom_threshold*abs(np.max(Pp)-np.min(Pp))))
p_arrivals = th[p_peaks]

plt.figure()
plt.plot(th, lin2dB(Py), label='Standard MUSIC')
plt.plot(th, lin2dB(Pp), label='Covariance Difference MUSIC')
for a in y_arrivals:
    plt.vlines(a, *plt.ylim(), linestyle='--', color='k', label=f'{round(a, 2)} degrees')
for a in p_arrivals:
    plt.vlines(a, *plt.ylim(), linestyle='--', color='r', label=f"{round(a, 2)} degrees")
plt.title(f"Psuedospectrum under Toeplitz Rnn:\nTrue Angles {np.round(np.rad2deg(theta),2)}")
plt.xlabel("AoA [degrees]")
plt.ylabel("Spectrum Magnitude [dB]")
plt.grid()
plt.legend()
plt.show()
