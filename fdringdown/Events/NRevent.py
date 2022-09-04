import numpy as np

# The base class containing data processing functions
from .Event import EventClass

# Manipulating file paths
import os

# Window functions
from scipy.signal import get_window

# Interferometer class
from ..interferometer import interferometer


class NRevent(EventClass):
    
    def __init__(self, name, surrogate, q, chi1, chi2, M, dist_mpc, f_low=10, 
                 f_ref=None, fs=4096, start_time=-512, duration=1024,
                 inclination=0, phi_ref=0, ra=0, dec=0, psi=0, t_event=0, 
                 IFO_names=['H1','L1','V1'], IFO_time_delay=True, asd='O3'):
        """
        Initialize the class.
        """
        # A name for the event
        self.name = name
        
        # Surrogate and source properties
        self.surrogate = surrogate
        self.q = q
        self.chi1 = chi1
        self.chi2 = chi2
        self.M = M
        self.dist_mpc = dist_mpc
        self.inclination = inclination
        self.phi_ref = phi_ref
        
        # Frequency range and time array
        self.f_low = f_low
        self.f_ref = f_ref
        self.fs = fs
        self.dt = 1/fs
        time = np.linspace(
            start_time, start_time+duration, num=int(duration*fs), 
            endpoint=False)
        
        # Sky location, arrival time and interferometers
        self.ra = ra
        self.dec = dec
        self.psi = psi
        self.t_event = t_event
        self.IFO_names = IFO_names
        self.IFO_time_delay = IFO_time_delay
        self.asd = asd
        
        # Create a list of interferometer objects
        self.IFO_list = []
        
        for IFO_name in IFO_names:
            self.IFO_list.append(interferometer(IFO_name))
            
        # Evaluate surrogate
        # ------------------
            
        # The surrogate models
        import gwsurrogate as gws
        
        # Used to calculate the remnant properties
        import surfinBH as gwsremnant
        
        # Download the surrogate if it hasn't been already - this saves the 
        # surrogate .h5 file to the 
        # lib/python/site-packages/gwsurrogate/surrogate_downloads folder
        if surrogate not in dir(gws):
            gws.catalog.pull(surrogate)
        
        # Load the surrogate
        sur = gws.LoadSurrogate(surrogate)
        
        # Evaluate the surrogate with the provided parameters
        sur_time, sur_signal, dyn = sur(
            q=q, chiA0=chi1, chiB0=chi2, M=M, dist_mpc=dist_mpc, f_low=f_low, 
            f_ref=f_ref, dt=self.dt, inclination=inclination, phi_ref=phi_ref, 
            units='mks')
        
        # Signal processing
        # -----------------
        
        # We need to pad the surrogate waveform, sur_signal, such that it is 
        # the same length as time, and the zero time is in the correct place
        
        # The location of the peak in the surrogate waveform (which we want to
        # correspond to time zero)
        sur_peak_index = np.argmax(abs(sur_signal))
        
        # The location of the time nearest zero in the time array (which is
        # where we want to put the peak of the waveform)
        peak_index = np.argmin(abs(time))
        
        # The number of data points before and after the peak index in the 
        # sur_signal array
        sur_pre_peak_len = sur_peak_index
        sur_post_peak_len = len(sur_signal) - 1 - sur_peak_index
        
        # The number of data points before and after the peak index in the 
        # time array
        pre_peak_len = peak_index
        post_peak_len = len(time) - 1 - peak_index
        
        # The lengths we need to pad the sur_signal array
        pre_pad = pre_peak_len - sur_pre_peak_len
        post_pad = post_peak_len - sur_post_peak_len
        
        # Pad the array
        self.signal = np.pad(sur_signal, [pre_pad,post_pad])
        
        # Shift time to ensure the peak is at zero
        self.time = time - time[peak_index]
        
        # A corresponding frequencies array will be useful
        self.freqs = np.fft.rfftfreq(len(self.time), d=self.dt)
        
        # SMOOTH HERE?
        
        # Project signal onto the interferometers
        self.project_signal()
        
        # Create a noise instance for each interferometer and add to the 
        # projected signals
        self.create_data()
        
        # Remnant properties
        # ------------------
        
        if surrogate == 'NRSur7dq4':
            
            # We use the surfinBH package to calculate the remnant black hole 
            # properties. For systems with precession we make sure to use 
            # consistant values for the reference epoch.
            
            # If f_ref is not given, NRSur7dq4 sets it to be equal to f_low
            if self.f_ref == None:
                self.f_ref = self.f_low
            
            # Load the surrogate remnant
            surrem = gwsremnant.LoadFits('NRSur7dq4Remnant')
            
            # The remnant mass and 1-sigma error estimate
            self.Mf, self.Mf_err = surrem.mf(
                q, self.chi1, self.chi2, omega0=np.pi*self.f_ref)
            
            # The remnant spin and 1-sigma error estimate
            self.chif, self.chif_err = surrem.chif(
                q, self.chi1, self.chi2, omega0=np.pi*self.f_ref)
            self.chif_mag = np.linalg.norm(self.chif)
            
        elif surrogate == 'NRHybSur3dq8':
            
            # Load the surrogate remnant
            surrem = gwsremnant.LoadFits('NRSur3dq8Remnant')
            
            # The remnant mass and 1-sigma error estimate
            Mf, Mf_err = surrem.mf(q, self.chi1, self.chi2)
            
            # The remnant spin and 1-sigma error estimate
            self.chif, self.chif_err = surrem.chif(q, self.chi1, self.chi2)
            self.chif_mag = np.linalg.norm(self.chif)
            
        # Convert the mass into solar masses
        self.Mf = self.M*Mf
        self.Mf_err = self.M*Mf_err
            
        # Angular coordinates of the final spin vector
        chif_norm = self.chif/self.chif_mag
        self.thetaf = np.arccos(chif_norm[2])
        self.phif = np.arctan2(chif_norm[1], chif_norm[0])
        
        
    # def smooth_signal(self):
        
    #     # Smoothly truncate the beginning of the waveform
    #     w = get_window(('tukey', 0.05), len(analysis_time))
    #     w[len(w)//2:] = 1
    #     n1 = np.where(analysis_data['H1'] != 0)[0][0]
    #     w = np.roll(w, n1)
    #     w[:n1] = 0
        
    #     for IFO_name in event.IFO_names:
    #         analysis_data[IFO_name] *= w
        
        
    def project_signal(self):
        """
        Project the signal onto each operating interferometer.
        """
        # Dictionary to store the projected signals
        self.projected_signal = {}
        
        if self.IFO_time_delay:
            for IFO in self.IFO_list:
                self.projected_signal[IFO.name] = IFO.response(
                    self.signal, self.ra, self.dec, self.t_event, self.psi, 
                    self.dt)
                
        else:
            for IFO in self.IFO_list:
                self.projected_signal[IFO.name] = IFO.project(
                    self.signal, self.ra, self.dec, self.t_event, self.psi)
        
        
    def create_data(self):
        """
        Simulate the noise in each interferometer, and add to the projected 
        signal to create the data.
        """
        # The directory of this file (current working directory)
        self.cwd = os.path.abspath(os.path.dirname(__file__))
        
        # The directory we store the NR event data (data directory)
        self.dd = os.path.abspath(
            self.cwd + '/../data/NR_events/' + self.name)
        
        # Create the data directory if it doesn't exist
        if not os.path.isdir(self.cwd + '/../data/NR_events/'):
            os.mkdir(self.cwd + '/../data/NR_events/')
        if not os.path.isdir(self.dd):
            os.mkdir(self.dd)
        
        # Dictionary to store the noise for each IFO
        self.noise = {}
        
        # Dictionary to store the data for each IFO
        self.data = {}
        
        for IFO in self.IFO_list:
            
            # Create some data with random noise if it doesn't already exist
            if not os.path.isfile(self.dd + '/' + IFO.name + '_strain.dat'):
            
                # Load the requested amplitude spectral density and evaluate 
                # at our frequencies
                asd_func = IFO.asd(self.asd)
                asd = asd_func(self.freqs)
                
                # To create a random noise instance, we first create a noise
                # series for which the magnitude squared has unit variance, 
                # and then we scale by the ASD (see the generating_noise 
                # notebook in examples)
                noise_real = np.random.normal(size=len(asd))
                noise_imag = np.random.normal(size=len(asd))
                
                # Multiply by the ASD, with an addiitonal normalization factor
                asd_random = asd*(noise_real + 1j*noise_imag)/np.sqrt(2)
                
                # We can then inverse FFT this ASD to get a noise series
                self.noise[IFO.name] = np.sqrt(1/(2*self.dt))*np.fft.irfft(
                    asd_random, norm='ortho')
                
                # Add the noise to the projected signal to get the data
                self.data[IFO.name] = self.projected_signal[IFO.name] + \
                    self.noise[IFO.name]
                    
                # Save to file
                print(f'Saving {IFO.name} data for NR event {self.name}')
                np.savetxt(
                    self.dd + '/' + IFO.name + '_strain.dat', 
                    self.data[IFO.name])
            
            else:
                
                print(f'Loading {IFO.name} data for NR event {self.name}')
                
                # Load existing data if it exists
                self.data[IFO.name] = np.loadtxt(
                    self.dd + '/' + IFO.name + '_strain.dat')
                
                # Recover the noise
                self.noise[IFO.name] = self.data[IFO.name] - \
                    self.projected_signal[IFO.name]
                    
    def estimate_SNR(self, f_range=(20,500)):
        
        SNR_dict = {}
        
        mask = (self.freqs >= f_range[0]) & (self.freqs < f_range[-1])
        
        for IFO in self.IFO_list:
            signal_FD = self.dt*np.fft.rfft(self.projected_signal[IFO.name])
            asd = IFO.asd(self.asd)(self.freqs[mask])
            SNR_dict[IFO.name] = np.sqrt(
                4*np.trapz(abs(signal_FD[mask])**2/asd**2, self.freqs[mask])
                )
            
        return SNR_dict
            