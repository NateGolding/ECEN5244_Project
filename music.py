import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

import noise as ns

class ULA:
    """1-Dimensional Uniform Linear Array Description"""
    def __init__(self, Nsamples=1024, M=16, N=3,
                    theta=np.deg2rad([-15, 3, 8]), d=1/2):
        self.Nsamples = Nsamples
        self.M = M
        self.N = N
        self.theta = theta
        self.d = d

    def manifold(self):
        """Get ULA array manifold (A matrix)"""
        return np.exp(-1j*2*np.pi*self.d*np.sin(self.theta))**(np.arange(self.M).reshape(-1,1))

    def signals(self, stype='random', pwr=1):
        """Generate signals for ULA simulator

        @param[in] stype Signal type (random, QPSK)
        @param[in] pwr Signal power (scalar or array-like of shape (N,))

        @retval Signals - complex valued ndarray (N, Nsamples)
        """
        pwr = np.asarray(pwr)
        if(stype=='random'):
            return (pwr.reshape(-1,1)/np.sqrt(2))*np.random.randn(self.N,self.Nsamples) + \
                    (pwr.reshape(-1,1)*1j/np.sqrt(2))*np.random.randn(self.N,self.Nsamples)
        else:
            raise NotImplementedError("Only random signals are available for now")

    def AF(self, n=100):
        """Get ULA array factor

        @param[in] n Number of points in AF
        @retval Angles for Array Factor - real valued ndarray (n,)
        @retval Array factor - complex valued ndarray (n,)
        """
        # AF_i has shape (M,)
        # [AF_i(t) for t in th] has shape (n, M)
        # AF has shape (n,)
        th = np.linspace(-np.pi/2, np.pi/2, n)
        AF_i = lambda t : np.exp(-1j*2*np.pi*self.d*np.sin(t))**(np.arange(self.M))
        return th, (1/self.M)*np.sum([AF_i(t) for t in th], axis=-1)

    def plot_AF(self, n=100, polar=True):
        """Plot ULA Array Factor

        @param[in] n Number of points in AF
        @param[in] polar Polar plot (on/off)
        """
        th, AF = self.AF(n=n)
        if(polar):
            fig, ax = plt.subplots(subplot_kw={'projection' : 'polar'})
            ax.plot(th, np.abs(AF))
        else:
            fig, ax = plt.subplots() 
            ax.set_xlabel("Angle of Arrival [deg]")
            ax.set_ylabel("Magnitude [linear]")
            ax.plot(np.rad2deg(th), np.abs(AF))
        ax.set_title("ULA Array Factor\n" + \
                     f"Nsensors: {self.M}\n" + \
                     f"Separation: {self.d}*lambda")
        ax.grid(True)
        plt.show()
 
def dB2lin(db):
    return 10**(db/10)

def lin2dB(db):
    return 10*np.log10(db) 

def fourth_cumulant(X):
    """Compute 4th order cumulant matrix

    @todo Make this more efficient!

    @param[in] X Input data, shape is (Nfeatures x Nsnapshots)

    @retval 4th order cumulant matrix (Nfeatures^2 x Nfeatures^2)
    """
    N = X.shape[0]
    Cxx = np.empty((N*N, N*N), dtype=X.dtype)
    for n in range(N*N):
        for m in range(N*N):
            i,j,k,l = n//N,n%N,m//N,m%N
            Cxx[n,m] = np.mean(X[i]*X[j].conj()*X[k].conj()*X[l]) \
                            - np.mean(X[i]*X[j].conj())*np.mean(X[k].conj()*X[l]) \
                            - np.mean(X[i]*X[k].conj())*np.mean(X[j]*X[l].conj())
    return Cxx


def get_peaks(X, prom_threshold, Nmax=np.inf):
    """Get peaks of a vector using prominence

    @param[in] X vector to find peak in
    @param[in] prom_threshold Prominence threshold as a % of full scale
    @param[in] Nmax Maximum number of peaks to detect (default np.inf, no limit)

    @retval Peak indices sorted in descending prominence order
    @retval Peak widths corresponding to indices
    """
    fullscale = abs(np.max(X)-np.min(X))
    peaks, properties = ss.find_peaks(X, prominence=(prom_threshold*fullscale), width=0)
    idx = np.argsort(properties['prominences'])[::-1]
    peaks = peaks[idx]
    #widths = properties['right_bases'][idx] - properties['left_bases'][idx]
    return peaks[:min(peaks.size, Nmax)] #, widths[:min(widths.size,Nmax)]


def base_music(Ryy, Nsignals, d, max_peaks=np.inf, prom_threshold=0.01):
    """MUSIC base algorithm for standard and covariance differencing

    @param[in] Ryy Array covariance matrix (M,M)
    @param[in] Nsignals Presumed number of incident signals (used for subspace formulation)
    @param[in] d Atennna spacing in wavelengths
    @param[in] max_peaks Maximum number of peaks to search for
    @param[in] prom_threshold Prominence theshold as a % of full scale (default 1%)

    @retval MUSIC psuedospectrum
    @retval Angles in degrees (x-axis for psuedospectrum)
    @retval Peaks in psuedospectrum with prominence above prom_threshold (as indices)
    """
    M = Ryy.shape[0]
    # eigendecomp, noise subspace extraction
    Dy,Uy = np.linalg.eig(Ryy)
    Un = Uy[:,(M-Nsignals):]

    # compute MUSIC psuedospectrum
    Npts = 2048
    Py = np.empty(Npts)
    th = np.linspace(-np.pi/2, np.pi/2, Npts)
    for i in range(Npts):    
        a_th = np.exp(-1j*2*np.pi*d*np.sin(th[i]))**(np.arange(M).reshape(-1,1))
        Py[i] = 1/np.abs((a_th.conj().T)@Un@(Un.conj().T)@a_th).item()

    # find peaks in spectrum
    th = np.rad2deg(th)
    y_peaks = get_peaks(Py, prom_threshold, max_peaks)

    return Py, th, y_peaks

def plot_spectrum(th, P, peaks, ax=None, title='MUSIC', label=''):
    """Forms the MUSIC spectrum plot
    
    Must call plt.show() AFTER calling this function

    @param[in] th Angles of the spectrum (xaxis)
    @param[in] P Psuedosepctrum (yaxis)
    @param[in] peaks Peak indices in spectrum
    @param[in] ax Axis to plot on (optional)
    @param[in] title Title to display (optional)
    """
    if(ax is None):
        plt.plot(th, lin2dB(P), label=label)
        ylim = plt.ylim()
        for a in th[peaks]:
            plt.vlines(a, *ylim, linestyle='--', color='k', label=f'{round(a, 2)} degrees')
        plt.title(title)
        plt.xlabel("Angle of Arrival [deg]")
        plt.ylabel("Magnitude [dB]")
        plt.grid()
        plt.legend()
    else:
        ax.plot(th, lin2dB(P))
        ylim = ax.get_ylim()
        for a in th[peaks]:
            ax.vlines(a, *ylim, linestyle='--', color='k', label=f'{round(a, 2)} degrees')
        ax.set_title(title)
        ax.set_xlabel("Angle of Arrival [deg]")
        ax.set_ylabel("Magnitude [dB]")
        ax.grid()
        ax.legend()

def plot_spectrums(angles, spectrums, titles):
    """Plot multiple MUSIC spectrums

    @param[in] angles List of truth angles
    @param[in] spectrums List or tuple of spectrum tuples (th, P, peaks)
    @param[in] titles List of spectrum titles
    """
    Nspecs = len(spectrums)
    fig, ax = plt.subplots(Nspecs, 1)
    plt.suptitle(f"True Angles: {np.rad2deg(theta)}")
    for i in range(Nspecs):
        P, th, peaks = spectrums[i]
        plot_spectrum(th, P, peaks, ax=ax[i], title=titles[i])
    plt.show()

def plot_spectrum_polar(th, P, title="MUSIC"):
    fig, ax = plt.subplots(subplot_kw={'projection' : 'polar'})
    ax.plot(np.deg2rad(th), lin2dB(np.abs(P)))
    plt.show()

def plot_covariance_cumulant(X, name=''):
    """Plot covariance and 4th cumulant matrices side-by-side

    @param[in] X Input signal matrix (M,Nsamples)
    @param[in] name Name of the signal (str, to be used in title)
    """
    Rxx = (1/X.shape[-1])*(X @ X.conj().T)
    Cxx = fourth_cumulant(X)

    Drx, Urx = np.linalg.eig(Rxx)
    Dcx, Ucx = np.linalg.eig(Cxx)

    fig, ax = plt.subplots(2,2)

    ax[0,0].set_title(name + " Covariance Matrix")
    sns.heatmap(np.abs(Rxx), ax=ax[0,0], annot=False)
    ax[0,1].set_title(name + " Covariance Eigenvalues")
    sns.heatmap(np.abs(np.diag(Drx)), ax=ax[0,1], annot=False)

    ax[1,0].set_title(name + " Cumulant Matrix")
    sns.heatmap(np.abs(Cxx), ax=ax[1,0], annot=False)
    ax[1,1].set_title(name + " Cumulant Eigenvalues")
    sns.heatmap(np.abs(np.diag(Dcx)), ax=ax[1,1], annot=False)

    plt.show()

def plot_covariances(s, x, y, n):
    """Plot the covariance matrices for the simulation on heatmap

    Must call plt.show() AFTER calling this function

    @param[in] s Signals (N,Nsamples)
    @param[in] x Clean (no noise) received signals (M,Nsamples)
    @param[in] y Noisy received signals (M,Nsamples)
    @param[in] n Noise (M,Nsamples)

    @retval Signal autocovariance (Rss)
    @retval Noiseless array autocovariance (Rxx)
    @retval Array autocovariance (Ryy)
    @retval Noise autocovariance (Rnn)
    """
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

    return Rss, Rxx, Ryy, Rnn


if __name__ == "__main__":
    
    ## ULA configuration:
    # 3 signals -> 0, -15, 20 degrees
    # 8 sensors -> lambda/4 separation
    Nsamples = 2048
    M = 16
    N = 3
    theta = np.deg2rad([-15, 3, 8])
    d = (1/4)
    SNR_dB = 10
    cr_min = 0.1
    prom_threshold = 0.01        # pct full scale prominence threshold for Pmusic peak finding

    # generate array manifold
    A = np.exp(-1j*2*np.pi*d*np.sin(theta))**(np.arange(M).reshape(-1,1))
   
    # generate signals (uniform power of 1)
    s = (1/np.sqrt(2))*np.random.randn(N,Nsamples) + (1j/np.sqrt(2))*np.random.randn(N,Nsamples)
    #s = (1/np.sqrt(2))*np.random.choice([1+1j, -1+1j, 1-1j, -1-1j], size=(N,Nsamples))
 
    # generate noise
    SNR = dB2lin(SNR_dB)
    npwr = 1/SNR
    #Rnn = sc.linalg.toeplitz(np.linspace(npwr, 0.75, M))
    #Rnn = np.diag(np.random.uniform(npwr, 2*npwr, size=M))
    Rnn = npwr*np.eye(M)
    n = ns.structured_noise(Rnn, Nsamples)

    # generate sensor readings
    x = A@s
    y = A@s + n

    # run MUSIC algorithms
    std = music_standard(y, N, d)
    toep = music_toeplitz_difference(y, N, d)
    diag = music_diagonal_difference(y, N, d, rho=0.98)
#    cum = music_cumulants(y, N, d)

    specs = [
        std,
        toep,
#        diag,
#        cum,
    ]
    titles=[
        'Standard MUSIC',
        'Toeplitz Covariance Differencing MUSIC',
#        'Non-Uniform Diagonal Covariance Differencing MUSIC',
#        '4th Cumulant MUISC'
    ]

    plot_spectrums(theta, specs, titles)
