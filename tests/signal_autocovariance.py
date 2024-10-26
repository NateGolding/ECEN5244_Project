import numpy as np
import scipy as sc
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

N = 3           # Number of signals
M = 10          # Number of receivers
Npts = 1000     # Number of time samples
    
# generate signals
signal_pwrs = np.random.randint(5, 10, M)
signals = np.matrix([pwr*np.random.randn(Npts) for pwr in signal_pwrs])

# covariance estimate of signal matrix
Rss_est = (signals @ signals.H) / Npts

# Plot the heatmap with Seaborn
sns.heatmap(Rss_est, cmap='viridis', annot=False)
plt.title(f'Signal Autocovariance Estimate ({Npts} samples)')
plt.xlabel('Receiver m/M')
plt.ylabel('Receiver m/M')
plt.show()

Npts_tests = np.array([100, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000])
mse = np.empty(Npts_tests.size)

i = 0
for Npts in Npts_tests:

    # generate signals
    signal_pwrs = np.random.randint(5, 10, M)
    signals = np.matrix([(pwr/np.sqrt(2))*np.random.randn(Npts) + 1j*(pwr/np.sqrt(2))*np.random.randn(Npts) for pwr in signal_pwrs])

    # covariance estimate of signal matrix
    Rss_est = np.abs(signals @ signals.H) / Npts

    # Plot the heatmap with Seaborn
    #sns.heatmap(Rss_est, cmap='viridis', annot=False)
    #plt.title('Signal Autocovariance')
    #plt.xlabel('Receiver m/M')
    #plt.ylabel('Receiver m/M')
    #plt.show()

    est_pwr = np.diag(Rss_est)
    mse[i] = np.mean((est_pwr - signal_pwrs**2)**2)
    i += 1

plt.figure()
plt.semilogx(Npts_tests, mse)
plt.scatter(Npts_tests, mse)
plt.title("MSE of Signal Powers over Nsamples")
plt.xlabel("Nsamples [log]")
plt.ylabel("$\|(\hat{\sigma^2} - \sigma^2)^2\|^2$")
plt.grid()
plt.show()
