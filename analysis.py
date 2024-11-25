import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import noise
import music

from standard_music import *
from toeplitz_music import *
from diagonal_music import *
from cumulant_music import *

def dict2str(d):
    return '\n'.join([f"{k.title().replace('_', ' ')} : {v}" for k,v in d.items()])

def dB2lin(db):
    return 10**(db/10)

def lin2dB(db):
    return 10*np.log10(db) 

class MUSICAnalyzer():
    """MUSIC Algorithm Performance Analyzer"""

    def __init__(self, angles=np.deg2rad([-13, 3, 8]), Nsensors=16, spacing=(1/2)):
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

        # parameter label lookup table
        self.paramlabel_LUT = {
            'Nsamples'    :   "Nsamples",
            'snr_db_min'  :   "SNR dB Min",
            'snr_db_max'  :   "SNR dB Max",
            'pmin'        :   "p Min",
            'pmax'        :   "p Max",
        }

    def generate_array(self, Nsamples=1024, signalType='gaussian', RnnType='uniform_diagonal',
            snr_db_min=0, snr_db_max=3, pmin=0, pmax=1):
        """Generate array samples

        Array manifold is generated based on class parameters (number of sensors,
        incident angles, sensor separation). Signals are generated based on signalType
        to be complex-valued and unit variance. Noise is generated based on a prescribed
        covariance matrix type (RnnType) and relevant parameters, which may include some
        of snr_db_min/max and pmin/max.

        Available signalTypes are:
        - gaussian
        - qpsk

        Available RnnTypes are:
        - uniform_diagonal: uses snr_db_min
        - nonuniform_diagonal: uses snr_db_min/max
        - symmetric_toeplitz: uses snr_db_min (main diagonal) and pmin/max
        - symmetric_nontoeplitz: uses snr_db_min/max and pmin/max

        @param[in] Nsamples Number of samples per estimation
        @param[in] signalType Type of signal (gaussian or qpsk)
        @param[in] RnnType Type of noise covariance (see options for details)
        @param[in] snr_db_min Minimum SNR in dB
        @param[in] snr_db_max Maximum SNR in dB
        @param[in] pmin Minimum correlation coefficient
        @param[in] pmax Maximum correlation coefficient

        @retval A Array manifold, shaped (M,N)
        @retval s Signal vector, shaped (N,Nsamples)
        @retval x Noiseless sensor readings, shaped (M,Nsamples)
        @retval n Noise vector, shaped (M,Nsamples)
        @retval y Noisy sensor readings, shaped (M,Nsamples)
        """
        # generate array manifold
        A = np.exp(-1j*2*np.pi*self.d*np.sin(self.angles))**(np.arange(self.M).reshape(-1,1))
            
        # generate unit power signals
        if(signalType=='gaussian'):
            s = (1/np.sqrt(2))*(np.random.randn(self.N, Nsamples) \
                                + 1j*(np.random.randn(self.N, Nsamples)))
        elif(signalType=='qpsk'):
            s = (1/np.sqrt(2))*np.random.choice([1+1j,-1+1j,1-1j,-1-1j],
                                size=(self.N, Nsamples))
        else:
            raise Exception(f"[ERROR] Signal type {signalType} is not recognized")

        # generate noise
        if(RnnType == 'uniform_diagonal'):
            Rnn = noise.uniform_diagonal_cov(self.M, snr_db_min)
        elif(RnnType == 'nonuniform_diagonal'):
            Rnn = noise.nonuniform_diagonal_cov(self.M, snr_db_min, snr_db_max)
        elif(RnnType == 'symmetric_toeplitz'):
            Rnn = noise.symmetric_toeplitz_cov(self.M, snr_db_min, pmin, pmax)
        elif(RnnType == 'symmetric_nontoeplitz'):
            Rnn = noise.symmetric_nontoeplitz_cov(self.M, snr_db_min, snr_db_max, pmin, pmax)
        else:
            raise Exception(f"[ERROR] Rnn type {RnnType} is not recognized")

        n = noise.structured_noise(Rnn, nsamples=Nsamples)

        # generate sensor readings
        x = A@s
        y = A@s + n

        return A, s, x, n, y

    def metrics(self, snr_db, Nsamples, signalType='gaussian', Rnn='uniform_diagonal',
                        algorithm='standard', prom_threshold=0.01, plot=False):
        """Gather relevant metrics from MUSIC estimator vs Nsamples

        @param[in] Nsamples Number of samples per estimation (array-like)
        @param[in] snr_db Signal to noise ratio in decibels (array-like)
        @param[in] signalType Type of signal (gaussian or qpsk)
        @param[in] Rnn Noise covariance matrix (Nsensors,Nsensors)
        @param[in] algorithm MUSIC algorithm to analyze
        @param[in] prom_threshold Prominence threshold for peak finding as pct full scale
        @param[in] plot Plotting switch (on/off)

        @details Available algorithms are: 
            - standard
            - toeplitz_difference
            - diagonal_difference
            - cumulants

        @retval Expected angle error (across Nsignals) vs (snr_db, Nsamples)
        @retval Standard deviation of angle error (across Nsignals) vs (snr_db, Nsamples)
        """
        snr_db = np.asarray(snr_db)
        Nsamples = np.asarray(Nsamples)
        avg_err = np.empty((snr_db.size, Nsamples.size))
        std_err = np.empty((snr_db.size, Nsamples.size))

        for i in range(snr_db.size):
            print(f"SNR {round(snr_db[i], 2)}dB")
            _, avg_err[i], std_err[i] = self.metrics_vs_nsamples(Nsamples, snr_db[i], signalType=signalType, Rnn=Rnn,
                                                        algorithm=algorithm, prom_threshold=prom_threshold, plot=False)

        if(plot):
            X,Y = np.meshgrid(snr_db, Nsamples)

            plt.figure(1)
            plt.title(algorithm.replace("_", " ").title() + \
                        " Bearing Estimate Expected Error")
            plt.pcolor(X,Y,avg_err.T, cmap='viridis')
            plt.yscale('log')
            cbar = plt.colorbar()
            cbar.set_label("Error [deg]")
            plt.xlabel("SNR [dB]")
            plt.ylabel("Nsamples")
            
            plt.figure(2)
            plt.title(algorithm.replace("_", " ").title() + \
                        " Bearing Estimate Error Std Deviation")
            plt.pcolor(X,Y,std_err.T, cmap='viridis')
            plt.yscale('log')
            cbar = plt.colorbar()
            cbar.set_label("Error [deg]")
            plt.xlabel("SNR [dB]")
            plt.ylabel("Nsamples")

            plt.show()

        return avg_err, std_err

    def _metrics_vs_param_single(self, params, param_label, varied_labels=None):
        """Gather relevant metrics from MUSIC estimator vs a varied parameter

        Parameter dictionary is expected to consist of the following entries:
        - 'plot' : bool             Turn plotting on/off
        - 'Nsamples' : int          Number of samples to integrate over
        - 'signalType' : str        Signal type (gaussian or qpsk)
        - 'RnnType' : str           Covariance type (see generate_array() for options)
        - 'snr_db_min' : float      Minimum SNR in dB
        - 'snr_db_max' : float      Maximum SNR in dB
        - 'pmin' : float            Minimum correlation coefficient
        - 'pmax' : float            Maximum correlation coefficient
        - 'algorithm' : str         MUSIC algorithm to analyze
        - 'prom_threshold : float   Prominence for peak finding in pct of full scale
        - 'max_peaks' : int         Maximum number of peaks to resolve

        @param[in] params Dictionary of parameters for MUSIC (see details)
        @param[in] param_label Key into params dictionary over which MUSIC is analyzed
        @param[in] varied_labels Keys into params dictionary which also vary
 
        @retval Raw angle error vs params[param_label]
        @retval Expected angle error (across Nsignals) vs params[param_label]
        @retval Standard deviation of angle error (across Nsignals) vs params[param_label]
        """
        # parse input arguments
        plot = params['plot']
        plotscale = params['plotscale']
        algorithm = params['algorithm']
        arraylabels = ['Nsamples', 'signalType', 'RnnType', 'snr_db_min',
                            'snr_db_max', 'pmin', 'pmax']
        arraylabels.remove(param_label)
        if(varied_labels is not None):
            [arraylabels.remove(l) for l in varied_labels]
        algolabels = ['prom_threshold', 'max_peaks']

        arrayparams = {k : params[k] for k in arraylabels}
        algoparams = {k : params[k] for k in algolabels}

        param = params[param_label]
        Npeaks = np.empty(param.shape)
        err = np.empty((*param.shape, self.N))
        avg_err = np.empty(param.shape)
        std_err = np.empty(param.shape)

        # run algorithm for each param value
        for i in range(param.size):
            
            # progress bar
            nchars = 50
            nhashes = int(nchars*i/param.size + 1)
            sys.stdout.write("\033[K")
            print(f"{param[i]}: [" \
                    + nhashes*"#" + (nchars-nhashes-1)*"." + "]", end='\r')

            param_kw = {param_label : param[i]}
            if(varied_labels is not None):
                varied_kw = {l : params[l][i] for l in varied_labels}
                _, _, _, _, y = self.generate_array(**param_kw, **varied_kw, **arrayparams)
            else: 
                _, _, _, _, y = self.generate_array(**param_kw, **arrayparams)

            # run MUSIC algorithm
            if algorithm=='standard':
                Py, thy, ypeaks = music_standard(y, self.N, self.d, **algoparams)
            elif algorithm=='toeplitz_difference':
                Py, thy, ypeaks = music_toeplitz_difference(y, self.N, self.d, **algoparams)
            elif algorithm=='diagonal_difference':
                Py, thy, ypeaks = music_diagonal_difference(y, self.N, self.d, **algoparams)
            elif algorithm=='cumulants':
                Py, thy, ypeaks = music_cumulants(y, self.N, self.d, **algoparams)
            else:
                raise Exception(f"[ERROR] Algorithm {algorithm} is not recognized")

            # convert ot degrees
            thydeg = np.rad2deg(thy)

            # animated plotting
            if(plot):
                if(i==0):
                    plt.figure()
                plt.clf()
                plt.title(algorithm.replace("_", " ").title() + \
                            f" Psuedospectrum")
                plt.xlabel("Angle of Arrival [deg]")
                plt.ylabel("Magnitude [dB]")
                plt.plot(thydeg, lin2dB(np.abs(Py)))
                plt.vlines(self.anglesdeg, *plt.ylim(), linestyle='--', color='k')
                plt.vlines(thydeg[ypeaks], *plt.ylim(), linestyle='--', color='r')
                plt.grid()
                plt.pause(0.05)

            # compute # of detected peaks
            Npeaks[i] = ypeaks.size
            estimate = thydeg[ypeaks[:self.N]]

            # arrange truth angles to minimize error
            cost = np.abs(self.anglesdeg[:,None] - estimate[None,:])
            # if too few bearings are resolved, augment cost with
            # unrealistic angle error (anything >180 is impossible)
            if(cost.shape[1] < self.N):
                cost = np.hstack((cost, np.full((self.N, self.N-cost.shape[1]), 360)))

            rowidx, colidx = sc.optimize.linear_sum_assignment(cost)
            estimate = np.concatenate([estimate, np.full(self.N-estimate.size, np.nan)])

            ordered_angles = self.anglesdeg[rowidx]
            estimate = estimate[colidx]

            # compute expected error, std deviation of error
            err[i] = np.abs(ordered_angles - estimate)
            avg_err[i] = np.mean(err[i][~np.isnan(err[i])])
            std_err[i] = np.std(err[i][~np.isnan(err[i])])

            # compute peak resolution
            # TODO!! - use bases? peak to median ratio?

        print()   
        if(plot):
            paramlabel = self.paramlabel_LUT[param_label]
            fig = plt.figure()
            plt.title(algorithm.replace("_", " ").title() + \
                        f" Bearing Estimate Error vs {paramlabel}")
            plt.xlabel(paramlabel)
            plt.ylabel("Error [deg]")
            plt.plot(param, avg_err, label="Expected Error")
            plt.scatter(param, avg_err, marker='.', label="Expected Error")
            plt.plot(param, std_err, label="Error Standard Deviation")
            plt.scatter(param, std_err, marker='.', label="Error Standard Deviation")
            plt.xscale(plotscale)
            plt.grid()
            plt.legend()
            plt.show()

        return err, avg_err, std_err, # peak_resolution

    def _metrics_vs_param_multiple(self, params, param_label, varied_labels=None):
        """Gather relevant metrics from MUSIC estimator vs parameter for multiple algorithms

        Parameter dictionary is expected to consist of the following entries:
        - 'plot' : bool             Turn plotting on/off
        - 'Nsamples' : int          Number of samples to integrate over
        - 'signalType' : str        Signal type (gaussian or qpsk)
        - 'RnnType' : str           Covariance type (see generate_array() for options)
        - 'snr_db_min' : float      Minimum SNR in dB
        - 'snr_db_max' : float      Maximum SNR in dB
        - 'pmin' : float            Minimum correlation coefficient
        - 'pmax' : float            Maximum correlation coefficient
        - 'algorithm' : str         MUSIC algorithm to analyze
        - 'prom_threshold : float   Prominence for peak finding in pct of full scale
        - 'max_peaks' : int         Maximum number of peaks to resolve

        @param[in] params Dictionary of parameters for MUSIC (see details)
        @param[in] param_label Key into params dictionary over which MUSIC is analyzed
        @param[in] varied_labels Keys into params dictionary which also vary
 
        @retval Raw angle error vs params[param_label]
        @retval Expected angle error (across Nsignals) vs params[param_label]
        @retval Standard deviation of angle error (across Nsignals) vs params[param_label]
        """
        avg_err = dict()
        std_err = dict()

        plot = params['plot']
        plotscale = params['plotscale']
        param = np.asarray(params[param_label])
        algorithms = np.asarray(params['algorithm'])
        for algo in algorithms:
            params['algorithm'] = algo
            params['plot'] = False
            _, mu, sigma = self._metrics_vs_param_single(params, param_label, varied_labels)
            avg_err[algo] = mu
            std_err[algo] = sigma
        
        if(plot): 
            paramlabel = self.paramlabel_LUT[param_label]
            fig1 = plt.figure(1)
            plt.title(f"Bearing Estimate Expected Error vs {paramlabel}")
            plt.xlabel(f"{paramlabel}")
            plt.ylabel("Expected Error [deg]")
            for algo in algorithms:
                plt.plot(param, avg_err[algo], label=algo.title().replace('_', ' '))
                plt.scatter(param, avg_err[algo], marker='.', label=algo.title().replace('_', ' '))
            plt.xscale(plotscale)
            plt.grid()
            plt.legend()

            fig2 = plt.figure(2)
            plt.title(f"Bearing Estimate Error Standard Deviation vs {paramlabel}")
            plt.xlabel(f"{paramlabel}")
            plt.ylabel("Error Standard Deviation [deg]")
            for algo in algorithms:
                plt.plot(param, std_err[algo], label=algo.title().replace('_', ' '))
                plt.scatter(param, std_err[algo], marker='.', label=algo.title().replace('_', ' '))
            plt.xscale(plotscale)
            plt.grid()
            plt.legend()
            plt.show()

        return avg_err, std_err

    def metrics_vs_param(self, params, param_label, varied_labels=None):
        """Gather relevant metrics from MUSIC estimator vs parameter

        Parameter dictionary is expected to consist of the following entries:
        - 'plot' : bool             Turn plotting on/off
        - 'Nsamples' : int          Number of samples to integrate over
        - 'signalType' : str        Signal type (gaussian or qpsk)
        - 'RnnType' : str           Covariance type (see generate_array() for options)
        - 'snr_db_min' : float      Minimum SNR in dB
        - 'snr_db_max' : float      Maximum SNR in dB
        - 'pmin' : float            Minimum correlation coefficient
        - 'pmax' : float            Maximum correlation coefficient
        - 'algorithm' : str         MUSIC algorithm to analyze
        - 'prom_threshold : float   Prominence for peak finding in pct of full scale
        - 'max_peaks' : int         Maximum number of peaks to resolve

        The entries which can be used as a varied parameter
        (param_label or varied_labels) are
        - 'Nsamples'
        - 'snr_db_min'
        - 'snr_db_max'
        - 'pmin'
        - 'pmax'

        @param[in] params Dictionary of parameters for MUSIC (see details)
        @param[in] param_label Key into params dictionary over which MUSIC is analyzed
        @param[in] varied_labels Keys into params dictionary which also vary
 
        @retval Raw angle error vs params[param_label]
        @retval Expected angle error (across Nsignals) vs params[param_label]
        @retval Standard deviation of angle error (across Nsignals) vs params[param_label]
        """
        if(len(params['algorithm']) > 1):
            avg_err, std_err = self._metrics_vs_param_multiple(params, param_label, varied_labels)
        else:
            avg_err, std_err = self._metrics_vs_param_single(params, param_label, varied_labels)
        return avg_err, std_err 



if __name__ == "__main__":

    #Nsamples = 1024
    Nsamples = np.logspace(1, 5, 20, dtype='int')
    pmin = 0.5
    pspread = 0.25
    #snr_db_min = np.linspace(-30, 20, 50)
    snr_db_min = 0
    snr_db_spread = 10

    params = {
        'plot' : True,
        'plotscale' : 'log',
        'Nsamples' : Nsamples,
        'signalType' : 'gaussian',
        'RnnType' : 'symmetric_toeplitz',
        'snr_db_min' : snr_db_min,
        'snr_db_max' : snr_db_min + snr_db_spread,
        'pmin' : pmin,
        'pmax' : pmin + pspread,
        'algorithm' : ['standard', 'toeplitz_difference', 'diagonal_difference', 'cumulants'],
        'prom_threshold' : 0.01,
        'max_peaks' : np.inf
    }

    an = MUSICAnalyzer()
    an.metrics_vs_param(params, 'Nsamples')
