import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

def gen_uc_rnd_signals(N, Npts, std_low=5, std_high=10):
    """Generate uncorrelated random signals

    @param N Number of uncorrelated signals
    @param Npts Number of time samples
    @param std_low Low-end of randomized signal std deviation (default 5)
    @param std_high High-end of randomized signal std deviation (default 10)

    @retval np.matrix of shape (N, Npts)
    """
    sigmas = np.random.randint(std_low, std_high, N)/np.sqrt(2)
    return np.matrix([sig*np.random.randn(Npts) + 1j*sig*np.random.randn(Npts) for sig in sigmas])


def gen_array_factor(N, M, d_low=1/32, d_high=1/4,
                            theta_low=-np.pi/2, theta_high=np.pi/2):
    """Generate array factor matrix A

    @param N Number of signals
    @param M Number of antennas
    @param d_low Minimum value of antenna separation, normalized to wavelength (default 1/32)
    @param d_high Maximum value of antenna separation, normalized to wavelength (default 1/4)
    @param theta_low Minimum value of AoA in rads (default -pi/2)
    @param theta_high Maximum value of AoA in rads (default pi/2)

    @retval Arrival A np.matrix of shape (M, N)
    @retval (AoA in rads, antenna separation)
    """
    theta = np.random.uniform(theta_low, theta_high, N)
    d = np.random.uniform(d_low, d_high, N)
    phi = d*np.sin(theta)
    return ((np.exp(1j*2*np.pi*phi).reshape(-1,1))**np.arange(M)).T, (theta, d)


N = 4           # Number of signals
M = 16          # Number of receivers
Npts = 2**14    # Number of time samples

# generate music formatted data (uncorrelated)
s = gen_uc_rnd_signals(N, Npts)
A, labels = gen_array_factor(N, M, d_low=1/16, d_high=1/16, theta_low=-np.pi/4, theta_high=np.pi/4)
n = gen_uc_rnd_signals(M, Npts)
x = A@s + n

## compute autocovariance estimate
Rxx = np.abs(x@x.H) / x.shape[-1]

sns.heatmap(Rxx, cmap='viridis', annot=False)
plt.title('Recieved Signal Autocovariance (Rxx)')
plt.xlabel('Receiver m/M')
plt.ylabel('Receiver m/M')
plt.show()

## decompose autocovariance estimate
J, E = np.linalg.eig(Rxx)
idx = np.argsort(J)[::-1]
J = J[idx]
E = E[idx]

plt.figure()
plt.title("Autocovariance Eigenvalues")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.scatter(range(J.size), J)
plt.plot(J)
plt.vlines([N+1], *plt.ylim(), linestyle='--', color='k')
plt.grid()
plt.show()

# obtain the noise eigenspace
E_noise = E[:, N:]

# find orthogonal directions ot noise subspace
Nspectrum =  1024
theta = np.linspace(-np.pi/2, np.pi/2, Nspectrum)
Pmusic = np.empty(Nspectrum)
for i in range(Nspectrum):
    a = np.exp(1j*2*np.pi*np.sin(theta[i])*1/4)**np.arange(M)
    Pmusic[i] = 1/np.abs(np.conj(a.T) @ E_noise @ E_noise.H @ a)

print(labels[0])
plt.figure()
plt.plot(theta, 10*np.log10(Pmusic))
plt.grid()
plt.show()

import sys
sys.exit(0)

## find orthogonal directions to noise subspace
# Nspectrum = 1024
# theta = np.linspace(-np.pi, np.pi, Nspectrum)
# d = np.linspace(0, 1/2, Nspectrum)
# X, Y = np.meshgrid(theta, d)
# 
# Pmusic = np.empty((Nspectrum, Nspectrum))
# for i in range(Nspectrum):
#     for j in range(Nspectrum):
#         a = np.array([np.exp(1j*2*np.pi*np.sin(theta[i])*d[j]*m) for m in np.arange(M)])
#         Pmusic[i,j] = 10*np.log10(1/np.abs(np.conj(a).T @ E_noise @ E_noise.H @ a)[0,0])
# 
# fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
# surf = ax.plot_surface(X, Y, Pmusic, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.xlabel("AoA $\theta$")
# plt.ylabel("Antenna Distance $d$")
# plt.show()
