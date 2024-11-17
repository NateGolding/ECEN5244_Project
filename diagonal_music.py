import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import noise as ns
from music import *

def music_diagonal_difference(y, Nsignals, d, rho=0.98, max_peaks=np.inf, prom_threshold=0.01):
    """Run covariance differencing with non-uniform diagonal assumption on received data

    @param[in] y Received data (y = As + n) of shape (# antennas, # samples)
    @param[in] Nsignals Presumed number of siganls
    @param[in] d Atennna spacing in wavelengths
    @param[in] max_peaks Maximum number of peaks to search for
    @param[in] prom_threshold Prominence theshold as a % of full scale (default 1%)

    @retval MUSIC psuedospectrum
    @retval Angles in degrees (x-axis for psuedospectrum)
    @retval Peaks in psuedospectrum with prominence above prom_threshold
    """
    M = y.shape[0]
    Nsamples = y.shape[-1]

    # compute array covariance, apply differencing
    Ryy = (1/Nsamples) * y@(y.conj().T)
    J = np.diag(rho**np.arange(M))
    Jinv = np.linalg.inv(J)
    Ryy = 1j*(J @ Ryy @ Jinv - Jinv @ Ryy @ J)

    return base_music(Ryy, Nsignals, d, max_peaks=max_peaks, prom_threshold=prom_threshold)
