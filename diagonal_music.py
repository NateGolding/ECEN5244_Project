import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import noise as ns
from music import *

def music_diagonal_difference(y, Nsignals, d, rho=0.6, max_peaks=np.inf, prom_threshold=0.01, plot=False):
    """Run covariance differencing with non-uniform diagonal assumption on received data

    Performs covariance differencing under a transformation matrix J using a diagonal Vandermonde
    structure with input variable rho (J = [1 rho rho^2 ... rho^M]). Note that when rho is complex
    valued, spurious bearings appear in the output spectrum.

    @param[in] y Received data (y = As + n) of shape (# antennas, # samples)
    @param[in] Nsignals Presumed number of siganls
    @param[in] d Atennna spacing in wavelengths
    @param[in] rho Rho value for differencing transform formulation (see details)
    @param[in] max_peaks Maximum number of peaks to search for
    @param[in] prom_threshold Prominence theshold as a % of full scale (default 1%)
    @param[in] plot Turn plotting on/off

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

    # eigendecomp, noise subspace extraction
    Dy,Uy = np.linalg.eig(Ryy)
    Un = Uy[:,(M-2*Nsignals):]
    
    if(plot):
        plt.figure()
        plt.title("Array Covariance Eigenvalue Matrix")
        sns.heatmap(np.diag(np.abs(Dy)), annot=True)
        plt.show()

    # compute MUSIC psuedospectrum
    Npts = 2048
    Py = np.empty(Npts)
    th = np.linspace(-np.pi/2, np.pi/2, Npts)

    for i in range(Npts): 
        a_th = np.exp(-1j*2*np.pi*d*np.sin(th[i]))**(np.arange(M).reshape(-1,1))
        py = (a_th.conj().T @ J.conj().T @ Un @ Un.conj().T @ J @ a_th) + \
                (a_th.conj().T @ Jinv.conj().T @ Un @ Un.conj().T @ Jinv @ a_th)
        Py[i] = 1/np.abs(py.item())

    # find peaks in spectrum
    y_peaks = get_peaks(Py, prom_threshold, max_peaks)
    
    if(plot):
        plt.figure()
        plt.suptitle(f"Incident Bearings {np.rad2deg(theta)}")
        plot_spectrum(th, Py, y_peaks, ax=None, title='Diagonal Differencing MUSIC', label='')
        plt.show()

    return Py, th, y_peaks


def _diagonal_vs_rho(ula, rho, Rnn=None, snr_db=3, plot=False):
    A = ula.manifold()
    s = ula.signals()
    n = ula.noise(Rnn=Rnn, snr_db=snr_db)
    x = A@s
    y = x + n
    
    rho = np.asarray(rho)
    Npeaks = np.empty(rho.shape)
    err = np.empty((*rho.shape, ula.N))
    avg_err = np.empty(rho.shape)
    std_err = np.empty(rho.shape)

    # run algorithm for each snr value
    for i in range(rho.size):
        
        # progress bar
        nchars = 50
        nhashes = int(nchars*i/rho.size + 1)
        sys.stdout.write("\033[K")
        print(f"rho: {rho[i]}: [" \
                + nhashes*"#" + (nchars-nhashes-1)*"." + "]", end='\r')

        # run MUSIC algorithm
        Py, thy, ypeaks = music_diagonal_difference(y, ula.N, ula.d, rho=rho[i], plot=False)

        # convert ot degrees
        thydeg = np.rad2deg(thy)

        # animated plotting
        if(plot):
            if(i==0):
                plt.figure()
            plt.clf()
            plt.title(f"Diagonal Differencing Psuedospectrum\nrho: {round(rho[i], 2)}")
            plt.xlabel("Angle of Arrival [deg]")
            plt.ylabel("Magnitude [dB]")
            plt.plot(thydeg, lin2dB(np.abs(Py)))
            plt.vlines(np.rad2deg(ula.theta), *plt.ylim(), linestyle='--', color='k')
            plt.vlines(thydeg[ypeaks], *plt.ylim(), linestyle='--', color='r')
            plt.grid()
            plt.pause(0.05)

        # compute # of detected peaks
        Npeaks[i] = ypeaks.size
        estimate = thydeg[ypeaks[:ula.N]]

        # arrange truth angles to minimize error
        cost = np.abs(np.rad2deg(ula.theta)[:,None] - estimate[None,:])
        # if too few bearings are resolved, augment cost with
        # unrealistic angle error (anything >180 is impossible)
        if(cost.shape[1] < ula.N):
            cost = np.hstack((cost, np.full((ula.N, ula.N-cost.shape[1]), 360)))

        rowidx, colidx = sc.optimize.linear_sum_assignment(cost)
        estimate = np.concatenate([estimate, np.full(ula.N-estimate.size, np.nan)])

        ordered_angles = np.rad2deg(ula.theta)[rowidx]
        estimate = estimate[colidx]

        # compute expected error, std deviation of error
        err[i] = np.abs(ordered_angles - estimate)
        avg_err[i] = np.mean(err[i][~np.isnan(err[i])])
        std_err[i] = np.std(err[i][~np.isnan(err[i])])

        # compute peak resolution
        # TODO!! - use bases? peak to median ratio?

    if(plot):
        fig = plt.figure()
        plt.title(f"Diagonal Differencing Bearing Estimate Error vs Rho" + \
                    f"\n{ula.Nsamples} Samples" + \
                    f"\n{snr_db}dB SNR")
        plt.xlabel("rho")
        plt.ylabel("Error [deg]")
        plt.plot(rho, avg_err, label="Expected Error")
        plt.scatter(rho, avg_err, marker='.', label="Expected Error")
        plt.plot(rho, std_err, label="Error Standard Deviation")
        plt.scatter(rho, std_err, marker='.', label="Error Standard Deviation")
        plt.grid()
        plt.legend()
        plt.show()

    return err, avg_err, std_err, # peak_resolution


def _diagonal_vs_rho_snr(ula, rho, snr_db, Rnn=None, plot=False):
    snr_db = np.asarray(snr_db)
    rho = np.asarray(rho)
    avg_err = np.empty((snr_db.size, rho.size))
    std_err = np.empty((snr_db.size, rho.size))

    for i in range(snr_db.size):
        print(f"SNR {round(snr_db[i], 2)}dB")
        _, avg_err[i], std_err[i] = _diagonal_vs_rho(ula, rho, Rnn=Rnn, snr_db=snr_db, plot=False)

    if(plot):
        X,Y = np.meshgrid(snr_db, rho)

        plt.figure(1)
        plt.title("Diagonal Differencing Bearing Estimate Expected Error")
        plt.pcolor(X,Y,np.log10(avg_err.T), cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label("Error [log-deg]")
        plt.xlabel("SNR [dB]")
        plt.ylabel("Rho")
        
        plt.figure(2)
        plt.title("Diagonal Differencing Bearing Estimate Error Std Deviation")
        plt.pcolor(X,Y,np.log10(std_err.T), cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label("Error [log-deg]")
        plt.xlabel("SNR [dB]")
        plt.ylabel("Rho")
        plt.show()

    return avg_err, std_err


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
    y = x

    rho = np.linspace(-0.99, 0.99, 20)
    snr_db = np.linspace(-30, 20, 20)
    _diagonal_vs_rho_snr(ula, rho, snr_db, Rnn=Rnn, plot=True)
    # compute music
    P, th, peaks = music_diagonal_difference(y, N, d, rho=0.6, plot=True)
