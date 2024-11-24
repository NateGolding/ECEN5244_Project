### NOTE: this does not run, it is purely textual code that used to belong
### to the MUSICAnalyzer() class

    def metrics_vs_snr(self, snr_db, snr_db_spread, Nsamples=1024, signalType='gaussian',
                        RnnType='uniform_diagonal', pmin=0, pmax=1, algorithm='standard',
                        prom_threshold=0.01, plot=False):
        """Gather relevant metrics from MUSIC estimator vs SNR

        @param[in] snr_db (Minimum) signal to noise ratio in decibels (array-like)
        @param[in] snr_db_spread Range of snr in dB (snr_db_max = snr_db + snr_db_spread)
        @param[in] Nsamples Number of samples per estimation
        @param[in] signalType Type of signal (gaussian or qpsk)

        @param[in] RnnType Noise covariance matrix (Nsensors,Nsensors)
        @param[in] pmin Minimum correlation coefficient
        @param[in] pmax Maximum correlation coefficient
 
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
            
            # progress bar
            nchars = 50
            nhashes = int(nchars*i/snr_db.size + 1)
            sys.stdout.write("\033[K")
            print(f"snr_db {snr_db[i]}: [" \
                    + nhashes*"#" + (nchars-nhashes-1)*"." + "]", end='\r')

            _, _, _, _, y = self.generate_array(Nsamples=Nsamples, signalType=signalType,
                                RnnType=Rnn,
                                snr_db_min=snr_db[i], 
                                snr_db_max=(snr_db[i] + snr_db_spread),
                                pmin=pmin, pmax=pmax)

            # run MUSIC algorithm
            if algorithm=='standard':
                Py, thy, ypeaks = music_standard(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='toeplitz_difference':
                Py, thy, ypeaks = music_toeplitz_difference(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='diagonal_difference':
                Py, thy, ypeaks = music_diagonal_difference(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='cumulants':
                Py, thy, ypeaks = music_cumulants(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
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
                            f" Psuedospectrum\nSNR: {round(snr_db[i], 2)}dB")
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
            fig = plt.figure()
            plt.title(algorithm.replace("_", " ").title() + \
                        " Bearing Estimate Error vs SNR" + \
                        f"\n{Nsamples} Samples")
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

    def metrics_vs_snr_multiple(self, snr_db, algorithms, Nsamples=1024,
                        signalType='gaussian', Rnn='uniform_diagonal', prom_threshold=0.01, plot=False):
        """Gather relevant metrics from MUSIC estimator vs SNR for multiple algorithms at once

        @param[in] snr_db Signal to noise ratio in decibels (array-like)
        @param[in] algorithm MUSIC algorithms to analyze (array-like)
        @param[in] Nsamples Number of samples per estimation
        @param[in] signalType Type of signal (gaussian or qpsk)
        @param[in] Rnn Noise covariance matrix (Nsensors,Nsensors)
        @param[in] prom_threshold Prominence threshold for peak finding as pct full scale
        @param[in] plot Plotting switch (on/off)

        @details Available algorithms are: 
            - standard
            - toeplitz_difference
            - diagonal_difference
            - cumulants

        @retval Dictionary of expected angle error (across Nsignals) vs snr_db
        @retval Dictionary of standard deviation of angle error (across Nsignals) vs snr_db
        """
        avg_err = dict()
        std_err = dict()

        algorithms = np.asarray(algorithms)
        for algo in algorithms:
            _, mu, sigma = self.metrics_vs_snr(snr_db, algorithm=algo, Nsamples=Nsamples,
                                    signalType=signalType, Rnn=Rnn,
                                    prom_threshold=prom_threshold, plot=False)
            avg_err[algo] = mu
            std_err[algo] = sigma

        if(plot): 
            fig1 = plt.figure(1)
            plt.title("Bearing Estimate Expected Error vs SNR" + \
                        f"\n{Nsamples} Samples")
            plt.xlabel("SNR [dB]")
            plt.ylabel("Expected Error [deg]")
            for algo in algorithms:
                plt.plot(snr_db, avg_err[algo], label=algo.title().replace('_', ' '))
                plt.scatter(snr_db, avg_err[algo], marker='.', label=algo.title().replace('_', ' '))
            plt.grid()
            plt.legend()

            fig2 = plt.figure(2)
            plt.title("Bearing Estimate Error Standard Deviation vs SNR" + \
                        f"\n{Nsamples} Samples")
            plt.xlabel("SNR [dB]")
            plt.ylabel("Error Standard Deviation [deg]")
            for algo in algorithms:
                plt.plot(snr_db, std_err[algo], label=algo.title().replace('_', ' '))
                plt.scatter(snr_db, std_err[algo], marker='.', label=algo.title().replace('_', ' '))
            plt.grid()
            plt.legend()
            plt.show()

        return avg_err, std_err
 
    def metrics_vs_nsamples(self, Nsamples, snr_db=3, signalType='gaussian', Rnn='uniform_diagonal',
                        algorithm='standard', prom_threshold=0.01, plot=False):
        """Gather relevant metrics from MUSIC estimator vs Nsamples

        @param[in] Nsamples Number of samples per estimation (array-like)
        @param[in] snr_db Signal to noise ratio in decibels
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

        @retval Raw angle error vs Nsamples
        @retval Expected angle error (across Nsignals) vs Nsamples
        @retval Standard deviation of angle error (across Nsignals) vs Nsamples
        """
        Nsamples = np.asarray(Nsamples)
        Npeaks = np.empty(Nsamples.shape)
        err = np.empty((*Nsamples.shape, self.N))
        avg_err = np.empty(Nsamples.shape)
        std_err = np.empty(Nsamples.shape)

        # run algorithm for each snr value
        for i in range(Nsamples.size):

            # progress bar
            nchars = 50
            nhashes = int(nchars*i/Nsamples.size + 1)
            sys.stdout.write("\033[K")
            print(f"Nsamples {Nsamples[i]}: [" \
                    + nhashes*"#" + (nchars-nhashes-1)*"." + "]", end='\r')
            _, _, _, _, y = self.generate_array(Nsamples[i], signalType, Rnn, snr_db)

            # run MUSIC algorithm
            if algorithm=='standard':
                Py, thy, ypeaks = music_standard(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='toeplitz_difference':
                Py, thy, ypeaks = music_toeplitz_difference(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='diagonal_difference':
                Py, thy, ypeaks = music_diagonal_difference(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            elif algorithm=='cumulants':
                Py, thy, ypeaks = music_cumulants(y, self.N, self.d,
                                    max_peaks=np.inf, prom_threshold=prom_threshold)
            else:
                raise Exception(f"[ERROR] Algorithm {algorithm} is not recognized")

            # convert to degrees
            thydeg = np.rad2deg(thy)

            # animated plotting
            if(plot):
                if(i==0):
                    plt.figure()
                plt.clf()
                plt.title(algorithm.replace("_", " ").title() + \
                            f" Psuedospectrum\nSNR: {Nsamples[i]} samples")
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
            fig = plt.figure()
            plt.title(algorithm.replace("_", " ").title() + \
                        " Bearing Estimate Error vs Nsamples" + \
                        f"\nSNR {round(snr_db,2)}dB")
            plt.xlabel("Nsamples [log10]")
            plt.ylabel("Error [deg]")
            plt.plot(np.log10(Nsamples), avg_err, label="Expected Error")
            plt.scatter(np.log10(Nsamples), avg_err, marker='.', label="Expected Error")
            plt.plot(np.log10(Nsamples), std_err, label="Error Standard Deviation")
            plt.scatter(np.log10(Nsamples), std_err, marker='.', label="Error Standard Deviation")
            plt.grid()
            plt.legend()
            plt.show()

        return err, avg_err, std_err, # peak_resolution

    def metrics_vs_nsamples_multiple(self, Nsamples, algorithms, snr_db=3,
            signalType='gaussian', Rnn='uniform_diagonal', prom_threshold=0.01, plot=False):
        """Gather relevant metrics from MUSIC estimator vs Nsamples

        @param[in] Nsamples Number of samples per estimation (array-like)
        @param[in] algorithms MUSIC algorithm to analyze (array-like)
        @param[in] snr_db Signal to noise ratio in decibels
        @param[in] signalType Type of signal (gaussian or qpsk)
        @param[in] Rnn Noise covariance matrix (Nsensors,Nsensors)
        @param[in] prom_threshold Prominence threshold for peak finding as pct full scale
        @param[in] plot Plotting switch (on/off)

        @details Available algorithms are: 
            - standard
            - toeplitz_difference
            - diagonal_difference
            - cumulants

        @retval Dictionary of expected angle error (across Nsignals) vs snr_db
        @retval Dictionary of standard deviation of angle error (across Nsignals) vs snr_db
        """
        avg_err = dict()
        std_err = dict()

        algorithms = np.asarray(algorithms)
        for algo in algorithms:
            _, mu, sigma = self.metrics_vs_nsamples(Nsamples, algorithm=algo, snr_db=snr_db,
                                    signalType=signalType, Rnn=Rnn,
                                    prom_threshold=prom_threshold, plot=False)
            avg_err[algo] = mu
            std_err[algo] = sigma

        if(plot): 
            fig1 = plt.figure(1)
            plt.title("Bearing Estimate Expected Error vs Nsamples" + \
                        f"\n{round(snr_db, 2)}dB SNR")
            plt.xlabel("Nsamples [-]")
            plt.ylabel("Expected Error [deg]")
            for algo in algorithms:
                plt.plot(Nsamples, avg_err[algo], label=algo.title().replace('_', ' '))
                plt.scatter(Nsamples, avg_err[algo], marker='.', label=algo.title().replace('_', ' '))
            plt.xscale('log')
            plt.grid()
            plt.legend()

            fig2 = plt.figure(2)
            plt.title("Bearing Estimate Error Standard Deviation vs Nsamples" + \
                        f"\n{round(snr_db, 2)}dB SNR")
            plt.xlabel("Nsamples [-]")
            plt.ylabel("Error Standard Deviation [deg]")
            for algo in algorithms:
                plt.plot(Nsamples, std_err[algo], label=algo.title().replace('_', ' '))
                plt.scatter(Nsamples, std_err[algo], marker='.', label=algo.title().replace('_', ' '))
            plt.xscale('log')
            plt.grid()
            plt.legend()
            plt.show()

        return avg_err, std_err
