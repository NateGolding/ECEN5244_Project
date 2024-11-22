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

def dB2lin(db):
    return 10**(db/10)

def lin2dB(db):
    return 10*np.log10(db) 

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

    return E @ (np.diag(np.sqrt(D))) @ n

def uniform_diagonal_cov(M, snr_db):
    """Create uniform diagonal noise covariance matrix

    @Note uses snr_db as if signal power is 1

    @param[in] M Number of sensors
    @param[in] snr_db Signal to noise ratio in dB

    @retval Noise covariance matrix shaped (M,M)
    """
    snr = dB2lin(snr_db)
    npwr = 1/snr
    return npwr*np.eye(M)

def nonuniform_diagonal_cov(M, snr_db_min, snr_db_max):
    """Create non-uniform diagonal noise covariance matrix

    Minimum and maximum noise variances are determined by the
    parameters snr_db_max and snr_db_min assuming all signal
    variances are 1. All other variances are uniformly sampled
    from [snr_db_min, snr_db_max]. The final results are randomly
    shuffled along the diagonal.

    @param[in] M Number of sensors
    @param[in] snr_db_min Minimum SNR in decibels
    @param[in] snr_db_max Maximum SNR in decibels

    @retval Noise covariance matrix (M,M)
    """
    nvar_min = 1/(dB2lin(snr_db_min))
    nvar_max = 1/(dB2lin(snr_db_max))
   
    Rnn = np.random.uniform(nvar_min, nvar_max, M)
    Rnn[:2] = [nvar_min, nvar_max]
    Rnn = np.random.permutation(Rnn)
    return np.diag(Rnn)

def block_diagonal_cov():
    raise NotImplementedError

def symmetric_toeplitz_cov(M, snr_db, pmin=0, pmax=1):
    """Create symmetric Toeplitz noise covariance matrix

    Noise variance (main diagonal) is determined by the snr_db
    parameter. The other diagonal elements are determined by
    the noise variance and correlation coefficients randomly
    sampled from [pmin, pmax] (default [0,1])

    @param[in] M Number of sensors
    @param[in] snr_db SNR in decibels
    @param[in] pmin Minimum correlation coefficient (default 0)
    @param[in] pmax Maximum correlation coefficient (default 1)

    @retval Noise covariance matrix (M,M)
    """
    nvar = 1/(dB2lin(snr_db)) 
    
    # correlation coefficient matrix
    p = np.random.uniform(pmin, pmax, size=M)
    p[0] = 1
    p = sc.linalg.toeplitz(p)

    # covariance matrix
    return nvar * p

def symmetric_nontoeplitz_cov(M, snr_db_min, snr_db_max, pmin=0, pmax=1):
    """Create symmetric non-Toeplitz noice covariance matrix

    Minimum and maximum noise variances are determined by the
    parameters snr_db_max and snr_db_min assuming all signal
    variances are 1. All other variances are uniformly sampled
    from [snr_db_min, snr_db_max]. The final results are shuffled
    randomly along the diagonal. The rest of the matrix is filled
    with a triangular matrix with values determined by the
    noise variance and correlation coefficients randomly sampled
    from [pmin, pmax], which is mirrored to maintain symmetry
    
    @param[in] M Number of sensors
    @param[in] snr_db_min Minimum SNR in decibels
    @param[in] snr_db_max Maximum SNR in decibels
    @param[in] pmin Minimum correlation coefficient (default 0)
    @param[in] pmax Maximum correlation coefficient (default 1)

    @retval Noise covariance matrix (M,M)
    """
    nvar_min = 1/(dB2lin(snr_db_min))
    nstd_min = np.sqrt(nvar_min)
    
    nvar_max = 1/(dB2lin(snr_db_max))
    nstd_max = np.sqrt(nvar_max)
    
    # correlation coefficient matrix
    p = np.random.uniform(pmin, pmax, size=(M,M))
    p = np.triu(p) + np.triu(p).T
    p[np.diag_indices(M)] = 1

    # covariance matrix
    Rnn = np.random.uniform(nstd_min, nstd_max, size=M)     # random values in [min, max]
    Rnn[:2] = [nstd_min, nstd_max]                          # ensure min/max are present
    Rnn = np.random.permutation(Rnn)                        # shuffle values
    Rnn = Rnn.reshape((-1, 1)) * Rnn                        # (M,M) matrix of pairwise stds
    Rnn = Rnn * p

    return Rnn * p


if __name__ == "__main__":

    # test structureed noise
    Rnn = nonuniform_diagonal_cov(3, 0, 3)

    n = structured_noise(Rnn, nsamples=1024)
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
