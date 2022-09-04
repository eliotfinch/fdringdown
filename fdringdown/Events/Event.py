import numpy as np

# Window functions
from scipy.signal import get_window

# Power spectrum estimation
from scipy.signal import welch

# Data filtering
from scipy.signal import butter, sosfilt

# Data resampling
from scipy.signal import resample

# 1D interpolation
from scipy.interpolate import interp1d

# Function to whiten data
from ..utils import whiten


class EventClass:
    """
    A base class which contains methods used by the NRevent and GWevent 
    classes.
    """
    
    def get_data_segment(self, time, data, start_time, segment_length, 
                         align_params=None, window=None):
        """
        Obtain a segment of data from each interferometer.
        
        An optional align_params dictionary allows the data from each
        interferometer to be aligned with the data from a reference 
        interferometer (see the align_params description below). 
        
        An optional window parameter allows the obtained data segment to have 
        a window applied. The window argument is passed to the 
        scipy.signal.get_window function.

        Parameters
        ----------
        time : array
            The time samples corresponding to the provided data. The full,
            default time array is stored as .time.
            
        data : dict
            The data to obtain the segments from. This should be a dictionary
            with keys corresponding to the interferometer names. Usually for
            analysis this will be the raw, noisy data (stored as .data).
            
        start_time : float
            The start time of the window. If this is equal to a an entry in
            the time array, that entry is included.
            
        segment_length : int
            The number of data samples to include in the segment. For example,
            to obtain 1 second of data, this should be 1*event.fs.
            
        align_params : dict, optional
            A dictionary containing the information needed to align the data
            from each interferometer. Required keys are:
            
             - ref_IFO : str
                 The name of the interferometer which we use as the reference
                 frame. No shifting is applied to the data of the reference
                 interferomter. The segment time array corresponds to the
                 reference interferometer times.
                 
             - right_ascension : float
                 The right ascension of the signal in radians.
            
             - declination : float
                 The declination of the signal in radians.
            
             - event_time : float
                 The GPS time of arrival of the signal.
            
            The default is None, in which case no data shifting is performed.
            
        window : string, float, or tuple, optional
            The type of window to use. See the scipy.signal.get_window
            documentation for details.

        Returns
        -------
        time_segment : array
            The times for the data segment.
            
        data_segment : dict
            The segments of data for each interferomter.
        """
        # Dictionary to store the segments from each interferometer
        data_segment = {}
        
        # Construct the detector_dt dictionary
        # ------------------------------------
        
        if align_params==None:
            # If no alignment parameters are given then no data shifting is 
            # performed, and we fill the detector_dt dictionary with zeros
            detector_dt = {}
            for IFO_name in self.IFO_names:
                detector_dt[IFO_name] = 0

        else:
            # Identify the reference IFO, and create the detector_dt 
            # dictionary for the given sky location
            for IFO in self.IFO_list:
                if IFO.name == align_params['ref_IFO']:
                    detector_dt = IFO.detector_dt_from_sky_loc(
                        self.IFO_list, 
                        align_params['right_ascension'], 
                        align_params['declination'], 
                        align_params['event_time'])
            
        # Mask data
        # ---------
        
        # Get the requested data segment for each interferometer
        for IFO_name in self.IFO_names:
            
            # The data shift
            dt = detector_dt[IFO_name]
            
            # Mask the requested length of data from the start time
            masked_data = data[IFO_name][time>=start_time-dt][:segment_length]
            
            # Window if requested
            if window is not None:
                masked_data *= get_window(window, len(masked_data))
                
            # Store to the dictionary
            data_segment[IFO_name] = masked_data
            
        # Mask time
        # ---------
                    
        # For the segment time array we use the time corresponding to the 
        # reference interferometer (dt=0)
        
        # Mask the requested length of time from the start time
        time_segment = time[time>=start_time][:segment_length]
            
        # Return the segment time and dictionary of data segments
        return time_segment, data_segment
    
    
    def estimate_asd(self, data, nperseg, noverlap, window='hann', 
                     average='mean', fs=None):
        """
        Estimate the amplitude spectral density (ASD) of some data using 
        Welch's method (see scipy.signal.welch). 

        Parameters
        ----------
        data : dict
            The data from which the ASD is estimated. This should be a 
            dictionary with keys corresponding to the interferometer names.
            
        nperseg : int
            The length of each segment used in Welch's method.
            
        noverlap : int
            Number of points to overlap between segments.
            
        window : string, float, or tuple, optional
            Desired window to use, as in get_data segment. The default is 
            'hann'.
            
        average : { 'mean', 'median' }, optional
            Method to use when averaging periodograms. Defaults to 'mean'. The 
            default is 'mean'.
            
        fs : int, optional
            The sampling frequency of the data. If None, the default sampling
            frequency is used (usually 4096Hz). The default is None.

        Returns
        -------
        asd_dict : dict
            The ASDs for each interferometer in the provided data dictionary.
        """
        # Dictionary to store the ASD functions
        asd_dict = {}
        
        # Check sampling frequency
        if fs is None:
            fs = self.fs
        
        # Estimate the ASD for each interferometer's data, and store to 
        # dictionary
        for IFO_name in self.IFO_names:
            
            # First, use SciPy Welch to estimate the power spectrum
            freqs, psd = welch(
                data[IFO_name], fs=fs, window=window, nperseg=nperseg, 
                noverlap=noverlap, average=average)
            
            # Square root to get the ASD, interpolate, and store to dictionary
            asd_dict[IFO_name] = interp1d(freqs, np.sqrt(psd))
            
        return asd_dict
    
    
    def whiten_data(self, data, asd=None, f_range=(32,512), dt=None):
        """
        Whiten some data with a ASD. See the whiten function in ringdown.utils 
        for more details.

        Parameters
        ----------
        data : dict
            The data to whiten. This should be a dictionary with keys 
            corresponding to the interferometer names.
            
        asd : dict, optional
            The ASD of the noise. This should be a dictionary with keys 
            corresponding to the interferometer names. If not provided, the 
            ASD associated with the event is used (this is applicable to NR
            events). The default is None.
            
        f_range : tuple, optional
            The lower and upper frequencies to consider in the data. We 
            effectively bandpass the data between these frequencies when
            whitening. The default is (32,512).
            
        dt : float, optional
            The time resolution of the data. If None, the default resolution 
            is used (usually 1/4096s). The default is None.

        Returns
        -------
        whitened_data : dict
            The whitened data for each interferometer.
        """
        # Dictionary to store the whitened data
        whitened_data = {}
        
        # Check time resolution
        if dt is None:
            dt = self.dt
        
        # Whiten the data for each interferometer and store to the dictionary
        for IFO in self.IFO_list:
            
            if asd==None:
                whitened_data[IFO.name] = whiten(
                    data[IFO.name], IFO.asd(self.asd), dt, f_range)
            
            else:
                whitened_data[IFO.name] = whiten(
                    data[IFO.name], asd[IFO.name], dt, f_range)
                
        return whitened_data
    
    
    def filter_data(self, data, N, Wn, btype='lowpass', fs=None):
        """
        Apply a Nth-order Butterworth filter to the data.

        Parameters
        ----------
        data : dict
            The data to filter. This should be a dictionary with keys 
            corresponding to the interferometer names.
            
        N : int
            The order of the filter.
            
        Wn : array_like
            The critical frequency or frequencies. For lowpass and highpass 
            filters, Wn is a scalar; for bandpass and bandstop filters, Wn is 
            a length-2 sequence.
            
        btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
            The type of filter. The default is 'lowpass'.
            
        fs : int, optional
            The sampling frequency of the data. If None, the default sampling
            frequency is used (usually 4096Hz). The default is None.

        Returns
        -------
        filetred_data : dict
            The filtered data for each interferometer.
        """
        # Dictionary to store the filtered data
        filtered_data = {}
        
        # Check sampling frequency
        if fs is None:
            fs = self.fs

        # Create filter
        sos = butter(N, Wn, btype=btype, fs=fs, output='sos')

        # Apply the filter to data from each interferometer and add to the
        # dictionary
        for IFO_name in self.IFO_names:
            filtered_data[IFO_name] = sosfilt(sos, data[IFO_name])
            
        return filtered_data
    
    
    def resample_data(self, time, data, fs, window=None):
        """
        Resample data from each interferometer.

        Parameters
        ----------
        time : array
            The time samples corresponding to the provided data. The full,
            default time array is stored as .time.
            
        data : dict
            The data to resample. This should be a dictionary with keys 
            corresponding to the interferometer names. Usually for analysis 
            this will be the raw, noisy data (stored as .data).
            
        fs : float
            The new sampling frequency, in Hz.
            
        window : string, float, or tuple, optional
            The type of window to use. See the scipy.signal.get_window
            documentation for details. The default is None.

        Returns
        -------
        resampled_time : array
            The new time array.
            
        resampled_data : dict
            The resampled data for each interferometer.
        """
        # Convert the new sampling frequency to a number of samples
        fs_old = 1/(time[1]-time[0])
        num = int(len(time)*(fs/fs_old))
        
        # Dictionary to store the resampled data
        resampled_data = {}
        
        for IFO_name in self.IFO_names:
            resampled_x, resampled_t = resample(data[IFO_name], num, t=time, window=window)
            resampled_data[IFO_name] = resampled_x
            
        return resampled_t, resampled_data