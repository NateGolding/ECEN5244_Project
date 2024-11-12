import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import noise
import music

def dB2lin(db):
    return 10**(db/10)

def lin2dB(db):
    return 10*np.log10(db) 

class MUSICAnalyzer():
    """MUSIC Algorithm Performance Analyzer"""

    def __init__(self, angles=np.deg2rad([-11, 3, 20]), Nsensors=16, spacing=(1/4)):
        """Initialize MUSIC analyzer

        @param[in] angles Angles of arrival in radians
        @param[in] Nsensors Number of sensor elements
        @param[in] spacing Sensor spacing in wavelengths
        """
        # ULA configuration
        self.angles = angles
        self.anglesdeg = np.rad2deg(angles)
        self.N = self.angles.size
        self.M = Nsensors
        self.d = spacing

    def generate_array(self, Nsamples=1024, signal_type='gaussian', Rnn=None, snr_db=3):
        """Generate a single-snapshot array scenario

        @param[in] Nsamples Number of samples per estimation
        @param[in] signal_type Type of signal (gaussian or qpsk)
        @param[in] Rnn Noise covariance matrix (Nsensors,Nsensors)
        @param[in] snr_db Signal to noise ratio in decibels

        @retval A Array manifold, shaped (M,N)
        @retval s Signal vector, shaped (N,Nsamples)
        @retval x Noiseless sensor readings, shaped (M,Nsamples)
        @retval n Noise vector, shaped (M,Nsamples)
        @retval y Noisy sensor readings, shaped (M,Nsamples)
        """
        # generate array manifold
        A = np.exp(-1j*2*np.pi*self.d*np.sin(self.angles))**(np.arange(self.M).reshape(-1,1))
            
        # generate unit power signals
        if(signal_type=='gaussian'):
            s = (1/np.sqrt(2))*(np.random.randn(self.N, Nsamples) \
                                + 1j*(np.random.randn(self.N, Nsamples)))
        elif(signal_type=='qpsk'):
            s = (1/np.sqrt(2))*np.random.choice([1+1j,-1+1j,1-1j,-1-1j],
                                size=(self.N, Nsamples))
        else:
            raise Exception(f"[ERROR] Signal type {signal_type} is not recognized")

        # generate noise
        snr = dB2lin(snr_db)
        npwr = 1/snr
        Rnn = npwr*np.eye(self.M) if(Rnn is None) else Rnn
        n = noise.structured_noise(Rnn, Nsamples)

        # generate sensor readings
        x = A@s
        y = A@s + n

        return A, s, x, n, y

    def metrics_vs_snr(self, snr_db, Nsamples=1024, signal_type='gaussian', Rnn=None,
                        algorithm='standard', prom_threshold=0.01, plot=False):
        """Gather relevant metrics from MUSIC estimator vs SNR

        @param[in] snr_db Signal to noise ratio in decibels (array-like)
        @param[in] Nsamples Number of samples per estimation
        @param[in] signal_type Type of signal (gaussian or qpsk)
        @param[in] Rnn Noise covariance matrix (Nsensors,Nsensors)
        @param[in] algorithm MUSIC algorithm to analyze
        @param[in] prom_threshold Prominence threshold for peak finding as pct full scale
        @param[in] plot Plotting switch (on/off)

        @details Available algorithms are: 
            - standard
            - toeplitz_difference
            - diagonal_difference
            - cumulants

        @retval Raw angle error vs snr_db
        @retval Expected angle error (across Nsignals) vs snr_db
        @retval Standard deviation of angle error (across Nsignals) vs snr_db
        """
        snr_db = np.asarray(snr_db)
        Npeaks = np.empty(snr_db.shape)
        err = np.empty((*snr_db.shape, self.N))
        avg_err = np.empty(snr_db.shape)
        std_err = np.empty(snr_db.shape)

        # run algorithm for each snr value
        for i in range(snr_db.size):
            _, _, _, _, y = self.generate_array(Nsamples, signal_type, Rnn, snr_db[i])

            # run MUSIC algorithm
            if algorithm=='standard':
                Py, thy, ypeaks = music.music_standard(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='toeplitz_difference':
                Py, thy, ypeaks = music.music_toeplitz_difference(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='diagonal_difference':
                Py, thy, ypeaks = music.music_diagonal_difference(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='cumulants':
                Py, thy, ypeaks = music.music_cumulants(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            else:
                raise Exception(f"[ERROR] Algorithm {algorithm} is not recognized")

            # animated plotting
            if(plot):
                if(i==0):
                    plt.figure()
                plt.clf()
                plt.title(algorithm.replace("_", " ").title() + \
                            f" Psuedospectrum\nSNR: {round(snr_db[i], 2)}dB")
                plt.xlabel("Angle of Arrival [deg]")
                plt.ylabel("Magnitude [dB]")
                plt.plot(thy, lin2dB(np.abs(Py)))
                plt.vlines(self.anglesdeg, *plt.ylim(), linestyle='--', color='k')
                plt.vlines(thy[ypeaks], *plt.ylim(), linestyle='--', color='r')
                plt.grid()
                plt.pause(0.05)

            # compute # of detected peaks
            Npeaks[i] = ypeaks.size
            estimate = thy[ypeaks[:self.N]]

            # arrange truth angles to minimize error
            cost = np.abs(self.anglesdeg[:,None] - estimate[None,:])
            # if too few bearings are resolved, augment cost with
            # unrealistic angle error (anything >180 is impossible)
            if(cost.shape[1] < self.N):
                cost = np.hstack((cost, np.full((self.N, self.N-cost.shape[1]), 360)))

            rowidx, colidx = sc.optimize.linear_sum_assignment(cost)
            estimate = np.concatenate([estimate, np.full(self.N-estimate.size, np.nan)])
            print(rowidx, colidx, estimate)

            ordered_angles = self.anglesdeg[rowidx]
            estimate = estimate[colidx]

            # compute expected error, std deviation of error
            err[i] = np.abs(ordered_angles - estimate)
            avg_err[i] = np.mean(err[i][~np.isnan(err[i])])
            std_err[i] = np.std(err[i][~np.isnan(err[i])])

            # compute peak resolution
            # TODO!! - use bases? peak to median ratio?
   
        if(plot): 
            fig = plt.figure()
            plt.title(algorithm.replace("_", " ").title() + \
                        " Bearing Estimate Error vs SNR")
            plt.xlabel("SNR [dB]")
            plt.ylabel("Error [deg]")
            plt.plot(snr_db, avg_err, label="Expected Error")
            plt.scatter(snr_db, avg_err, marker='.', label="Expected Error")
            plt.plot(snr_db, std_err, label="Error Standard Deviation")
            plt.scatter(snr_db, std_err, marker='.', label="Error Standard Deviation")
            plt.grid()
            plt.legend()
            plt.show()

        return err, avg_err, std_err, # peak_resolution




if __name__ == "__main__":

    an = MUSICAnalyzer()
    snr_db = np.linspace(-30, 20, 100)
    #err, avg, std = an.metrics_vs_snr(snr_db, algorithm='standard', plot=True)
    #err, avg, std = an.metrics_vs_snr(snr_db, algorithm='toeplitz_difference', plot=True)
    #err, avg, std = an.metrics_vs_snr(snr_db, algorithm='diagonal_difference', plot=True)
    err, avg, std = an.metrics_vs_snr(snr_db, algorithm='cumulants', plot=True)
