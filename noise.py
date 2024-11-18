"""
@file noise.py
@brief Noise generator functions

Functions for generating various noise fields
including white, colored, Toeplitz, Block Diagonal,
etc...
"""
import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

def structured_noise(Rnn, nsamples=1, cmplx=True):
    """Generates noise samples of a prescribed covariance

    @param[in] Rnn Noise covariance matrix (MxM)
    @param[in] nsamples number of samples of noise distribution (default 1)
    @param[in] cmplx Complex-valued (on/off)

    @retval (Mxnsamples) ndarray of noise samples
    """
    # check matrix is square
    if((Rnn.ndim != 2) or (Rnn.shape[0] != Rnn.shape[1])):
        raise Exception(f"Input matrix Rnn is not square: shape {Rnn.shape}")
    # generate samples
    D, E = np.linalg.eig(Rnn)
    if(cmplx):
        n = (1/np.sqrt(2)) * (np.random.randn(Rnn.shape[0], nsamples) + 
                                1j*np.random.randn(Rnn.shape[0], nsamples))
    else:
        n = np.random.randn(Rnn.shape[0], nsamples)
    return E @ (np.diag(D)**0.5) @ n





if __name__ == "__main__":

    # test structureed noise
    Rnn = np.array([[1, 0.2, 0.01],
                    [0.2, 1, 0.3],
                    [0.01, 0.3, 1]])

    n = structured_noise(Rnn, nsamples=10000)
    Rnn_est = (1/n.shape[-1]) * n @ (n.conj().T)

    # show histogram
    fig, ax = plt.subplots(3,2)
    for i in range(3):
        rstd = np.std(np.real(n[i]))
        istd = np.std(np.imag(n[i]))
        ax[i,0].set_title(f"Sensor {i} Noise PDF (real)")
        ax[i,1].set_title(f"Sensor {i} Noise PDF (imag)")
        sns.histplot(np.real(n[i]), stat='probability', ax=ax[i,0])
        sns.histplot(np.imag(n[i]), stat='probability', ax=ax[i,1])
        ax[i,0].vlines([rstd, -rstd], *ax[i,0].get_ylim(), linestyle='--',
                        color='k', label=f"Std Dev: {round(rstd,3)}")
        ax[i,1].vlines([istd, -istd], *ax[i,0].get_ylim(), linestyle='--',
                        color='k', label=f"Std Dev: {round(istd,3)}")
        ax[i,0].legend()
        ax[i,1].legend()
    plt.show()

    # show
    fig, ax = plt.subplots(1,2)
    sns.heatmap(np.abs(Rnn), ax=ax[0], annot=True)
    ax[0].set_title("True Rnn")
    sns.heatmap(np.abs(Rnn_est), ax=ax[1], annot=True)
    ax[1].set_title("Generated Rnn")
    plt.show()
