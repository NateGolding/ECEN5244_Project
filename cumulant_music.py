import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import noise as ns
from music import *

def music_cumulants(y, Nsignals, d, max_peaks=np.inf, prom_threshold=0.01, plot=False):
    """Run 4th order cumulants method on recieved data

    @param[in] y Received data (y = As + n) of shape (# antennas, # samples)
    @param[in] Nsignals Presumed number of siganls
    @param[in] d Atennna spacing in wavelengths
    @param[in] max_peaks Maximum number of peaks to search for
    @param[in] prom_threshold Prominence theshold as a % of full scale (default 1%)
    @param[in] plot Turn plotting on/off

    @retval MUSIC psuedospectrum
    @retval Angles in radians (x-axis for psuedospectrum)
    @retval Peaks in psuedospectrum with prominence above prom_threshold
    """
    M = y.shape[0]
    Cyy = fourth_cumulant(y) 
    
    # eigendecomp, noise subspace extraction
    Dy,Uy = np.linalg.eig(Cyy)
    Un = Uy[:,(Cyy.shape[0] - Nsignals**2):]
    
    if(plot):
        plt.figure()
        plt.title("Array Cumulant Eigenvalue Matrix")
        sns.heatmap(np.diag(np.abs(Dy)), annot=False)
        plt.show()

    # compute MUSIC psuedospectrum
    Npts = 2048
    Py = np.empty(Npts)
    th = np.linspace(-np.pi/2, np.pi/2, Npts)
    for i in range(Npts):    
        a_th = np.exp(-1j*2*np.pi*d*np.sin(th[i]))**(np.arange(M).reshape(-1,1))
        w = np.kron(a_th, a_th.conj()).conj().T @ Un
        Py[i] = 1/np.abs(w @ w.conj().T).item()

    y_peaks = get_peaks(Py, prom_threshold, max_peaks)
    
    # handle spurious bearings
    bearings = th[y_peaks]
    A12 = np.exp(-1j*2*np.pi*d*np.sin(bearings))**(np.arange(M).reshape(-1,1))
    A12_H = A12.conj().T
    invterm = np.linalg.inv(A12_H@A12)
    Rss12 = np.diag(invterm @ A12_H @ Ryy @ A12 @ invterm)
    idx = np.argsort(Rss12)[::-1]
    y_peaks = (y_peaks[idx])[:Nsignals]
    
    if(plot):
        plt.figure()
        plt.suptitle(f"Incident Bearings {np.rad2deg(theta)}")
        plot_spectrum(th, Py, y_peaks, ax=None, title='Cumulants MUSIC', label='')
        plt.show()

    return Py, th, y_peaks



if __name__ == "__main__":
    Nsamples = 1024
    M = 16
    N = 3
    theta = np.deg2rad([-13, 3, 8])
    d = 1/2
    snr_db = 20
   
    # generate ULA configuration 
    ula = ULA(Nsamples=Nsamples, M=M, theta=theta, d=d)
    A = ula.manifold()
    s = ula.signals()

    # generate noise
    SNR = dB2lin(snr_db)
    npwr = 1/SNR
    Rnn = npwr*np.eye(M)
    n = ns.structured_noise(Rnn, Nsamples)

    # generate sensor readings
    x = A@s
    y = x + n

    Ryy = (1/Nsamples) * y@(y.conj().T)
    J = np.eye(M,M)[::-1, :]
    Ryy = Ryy - J@Ryy@J

    # compute music
    P, th, peaks = music_cumulants(y, N, d, plot=True)
