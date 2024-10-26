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
    """
    theta = np.random.uniform(-np.pi/2, np.pi/2, N)
    d = np.random.uniform(1/32, 1/4, N)
    phi = d*np.sin(theta)
    return ((np.exp(1j*2*np.pi*phi).reshape(-1,1))**np.arange(M)).T
    
############################################################################################

Npts = 1000     # number of time samples
N = 4           # number of signals
M = 16          # number of antennas

## generate MUSIC formatted data
signals = gen_uc_rnd_signals(N, Npts)
A = gen_array_factor(N, M)
x = A @ signals + gen_uc_rnd_signals(M, Npts, std_low=0, std_high=2)
print(x.shape)
