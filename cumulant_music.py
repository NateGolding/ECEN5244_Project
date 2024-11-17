import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import noise as ns
from music import *

def music_cumulants(y, Nsignals, d, max_peaks=np.inf, prom_threshold=0.01):
    """Run 4th order cumulants method on recieved data

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
    Cyy = fourth_cumulant(y)
    
    # eigendecomp, noise subspace extraction
    Dy,Uy = np.linalg.eig(Cyy)
    Un = Uy[:,(Cyy.shape[0] - Nsignals**2):]

    # compute MUSIC psuedospectrum
    Npts = 2048
    Py = np.empty(Npts)
    th = np.linspace(-np.pi/2, np.pi/2, Npts)
    for i in range(Npts):    
        a_th = np.exp(-1j*2*np.pi*d*np.sin(th[i]))**(np.arange(M).reshape(-1,1))
        w = np.kron(a_th, a_th.conj()).conj().T @ Un
        Py[i] = 1/np.abs(w @ w.conj().T).item()
    #Py = Py[::-1]
    th = np.rad2deg(th)
    y_peaks = get_peaks(Py, prom_threshold, max_peaks)

    return Py, th, y_peaks
