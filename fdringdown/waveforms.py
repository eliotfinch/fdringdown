import numpy as np

from pykerr import spheroidal
from pykerr.qnm import qnmomega

from pycbc.waveform import get_td_waveform

from scipy.signal import get_window

from .utils import Msun, G, c, param_order, param_labels, latex_labels


class ringdown:
    """
    Class to construct a ringdown waveform.
    
    Parameters
    ----------
    modes : list
        A list of (l,m,n) tuples (where l is the angular number of the mode, m
        is the azimuthal number, and n is the overtone number) to include in 
        the ringdown signal. 
    """
    
    def __init__(self, modes):
        """
        Initialize the class.
        """
        self.modes = modes
        
        # The number of modes in this model
        self.Nmodes = len(modes)
        
        # The number of parameters in this model (a amplitude and phase for 
        # each mode, plus a inclination, azimuthal angle, start time, mass and 
        # spin)
        self.N_params = 2*self.Nmodes + 5
        
        # Factor for unit conversion
        self.conversion = Msun*(G/c**3)
        
        self.create_labels()
        
    def create_labels(self):
        """
        Create self.labels and self.latex_labels dictionaries, that contain 
        all the required parameters for the initialised model and their labels.
        Also creates lists of the labels (self.labels_list and 
        self.latex_labels_list) ordered according to param_order in utils.
        """
        # Parameter "types" used in this waveform (this should be a subset of 
        # the param_order list from utils)
        params = [
            'start_time',
            'rd_amplitudes',
            'rd_phases',
            'mass',
            'spin',
            'inclination',
            'azimuth'
            ]
        
        # Label dictionary: keys are the above param types, values are a list
        # of shorter param labels
        self.labels = {}
        for param_type in params:
            label = param_labels[param_type]
            if '{}' in label:
                self.labels[param_type] = [label.format(n) for n in range(self.Nmodes)]
            else:
                self.labels[param_type] = [label]
        
        # Label list: ordered list of all param labels used in this waveform
        self.labels_list = []
        for param_type in param_order:
            if param_type in params:
                self.labels_list += self.labels[param_type]
                
        # LaTeX label dictionary: same as self.labels, but values are a list
        # of LaTeX labels
        self.latex_labels = {}
        for param_type in params:
            label = latex_labels[param_type]
            if '{}' in label:
                self.latex_labels[param_type] = [label.format(n) for n in range(self.Nmodes)]
            else:
                self.latex_labels[param_type] = [label]
            
        # LaTeX label list: ordered list of all LaTeX param labels used in 
        # this waveform
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in params:
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, times, parameters):
        """
        Construct a ringdown waveform consisting of a sum of QNMs. Currently 
        this neglects retrograde modes, and also assumes perturbations 
        symmetric under equatorial relfections. See Eq. B2 in 
        https://arxiv.org/abs/2107.05609
        

        Parameters
        ----------
        time : array_like
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Inspect the class 
            .labels to see the required keys.

        Returns
        -------
        ndarray
            The complex ringdown sum.
        """
        # Get the parameters from the dictionary
        start_time = parameters[self.labels['start_time'][0]]
        
        amplitudes = np.array(
            [parameters[n] for n in self.labels['rd_amplitudes']]
            )
            
        phases = np.array(
            [parameters[n] for n in self.labels['rd_phases']]
            )
            
        mass = parameters[self.labels['mass'][0]]*self.conversion
        spin = parameters[self.labels['spin'][0]]
        
        inclination = parameters[self.labels['inclination'][0]]
        azimuth = parameters[self.labels['azimuth'][0]]
        
        # Obtain the (complex) QNM frequency list
        frequencies = [qnmomega(spin, *mode)/mass for mode in self.modes]
        
        # Create an empty array to add the result to
        h = np.zeros(len(times), dtype=complex)
        
        # Mask so that we only consider times after the start time
        t_mask = times >= start_time
    
        # Shift the time so that the waveform starts at time zero, and mask 
        # times after the start time
        times = (times - start_time)[t_mask]
        
        # Evaluate the ringdown sum
        for i, (l,m,n) in enumerate(self.modes):
            h[t_mask] += amplitudes[i]*(
                
                spheroidal(inclination, spin, l, m, n, phi=azimuth)\
                *np.exp(-1j*(frequencies[i]*times-phases[i]))\
                    
                +np.conjugate(spheroidal(np.pi-inclination, spin, l, m, n, phi=azimuth))\
                *np.exp(1j*(np.conjugate(frequencies[i])*times-phases[i]))
                )
        
        return h


class wavelet_sum:
    """
    Class to construct a wavelet sum waveform.
    
    Parameters
    ----------
    Nwavelets : int
        The number of wavelets to include in the sum. Note that each wavelet 
        contains a positive frequency component and a negative frequency
        component (each with a amplitude and phase) for general ellipticity.
    """
    
    def __init__(self, Nwavelets):
        """
        Initialize the class.
        """
        self.Nwavelets = Nwavelets
        
        # The number of parameters in this model. For each wavelet we have:
        #   - a central time
        #   - two amplitudes
        #   - two phases
        #   - a frequency
        #   - a damping time
        self.N_params = 7*self.Nwavelets
        
        self.create_labels()
            
    def create_labels(self):
        """
        Create self.labels and self.latex_labels dictionaries, that contain 
        all the required parameters for the initialised model and their labels.
        Also creates lists of the labels (self.labels_list and 
        self.latex_labels_list) ordered according to param_order in utils.
        """
        # Parameter "types" used in this waveform (this should be a subset of 
        # the param_order list from utils)
        params = [
            'central_times',
            'w_plus_amplitudes',
            'w_minus_amplitudes',
            'w_plus_phases',
            'w_minus_phases',
            'frequencies',
            'damping_times',
            ]
        
        # Label dictionary: keys are the above param types, values are a list
        # of shorter param labels
        self.labels = {}
        for param_type in params:
            label = param_labels[param_type]
            if '{}' in label:
                self.labels[param_type] = [label.format(n) for n in range(self.Nwavelets)]
            else:
                self.labels[param_type] = [label]
        
        # Label list: ordered list of all param labels used in this waveform
        self.labels_list = []
        for param_type in param_order:
            if param_type in params:
                self.labels_list += self.labels[param_type]
                
        # LaTeX label dictionary: same as self.labels, but values are a list
        # of LaTeX labels
        self.latex_labels = {}
        for param_type in params:
            label = latex_labels[param_type]
            if '{}' in label:
                self.latex_labels[param_type] = [label.format(n) for n in range(self.Nwavelets)]
            else:
                self.latex_labels[param_type] = [label]
            
        # LaTeX label list: ordered list of all LaTeX param labels used in 
        # this waveform
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in params:
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, times, parameters):
        """
        Construct a sum of wavelets.

        Parameters
        ----------
        time : array_like
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Inspect the class 
            .labels to see the required keys.

        Returns
        -------
        ndarray
            The complex wavelet sum.
        """
        # Get required parameters from the dictionary
        central_times = np.array(
            [parameters[n] for n in self.labels['central_times']]
            )
        
        plus_amplitudes = np.array(
            [parameters[n] for n in self.labels['w_plus_amplitudes']]
            )
        
        minus_amplitudes = np.array(
            [parameters[n] for n in self.labels['w_minus_amplitudes']]
            )
        
        plus_phases = np.array(
            [parameters[n] for n in self.labels['w_plus_phases']]
            )
        
        minus_phases = np.array(
            [parameters[n] for n in self.labels['w_minus_phases']]
            )
            
        frequencies = np.array(
            [parameters[n] for n in self.labels['frequencies']]
            )
            
        damping_times = np.array(
            [parameters[n] for n in self.labels['damping_times']]
            )
        
        # Create an empty array to add the result to
        h = np.zeros(len(times), dtype=complex)
        
        # Evaluate the wavelet sum
        for eta, Ap, Am, phip, phim, nu, tau in zip(
                central_times,
                plus_amplitudes,
                minus_amplitudes,
                plus_phases,
                minus_phases,
                frequencies,
                damping_times
                ):
            
            h += Ap*np.exp(1j*phip)\
                *np.exp(-((times-eta)/tau)**2)\
                *np.exp(-1j*2*np.pi*nu*(times-eta))\
                + Am*np.exp(1j*phim)\
                *np.exp(-((times-eta)/tau)**2)\
                *np.exp(1j*2*np.pi*nu*(times-eta))
        
        return h
    

class imr:
    """
    Class to construct an IMR waveform. Currently this only works for 
    IMRPhenomD via PyCBC - can we make this work for other approximants?

    Parameters
    ----------
    """

    def __init__(self, approximant='IMRPhenomD', fs=4096., f_low=20, f_ref=20):
        """
        Initialize the class.
        """
        self.approximant = approximant
        self.fs = fs
        self.f_low = f_low
        self.f_ref = f_ref

        if approximant == 'IMRPhenomD':
            self.N_params = 8
        
        self.create_labels()
            
    def create_labels(self):
        """
        Create self.labels and self.latex_labels dictionaries, that contain 
        all the required parameters for the initialised model and their labels.
        Also creates lists of the labels (self.labels_list and 
        self.latex_labels_list) ordered according to param_order in utils.
        """
        # Parameter "types" used in this waveform (this should be a subset of 
        # the param_order list from utils)

        params = [
            'mass_1',
            'mass_2',
            'spin_1',
            'spin_2',
            'distance',
            'imr_inclination',
            'imr_phase',
            'peak_time',
            ]
        
        # Label dictionary: keys are the above param types, values are a list
        # of shorter param labels
        self.labels = {}
        for param_type in params:
            self.labels[param_type] = [param_labels[param_type]]
        
        # Label list: ordered list of all param labels used in this waveform
        self.labels_list = []
        for param_type in param_order:
            if param_type in params:
                self.labels_list += self.labels[param_type]
                
        # LaTeX label dictionary: same as self.labels, but values are a list
        # of LaTeX labels
        self.latex_labels = {}
        for param_type in params:
            self.latex_labels[param_type] = [latex_labels[param_type]]
            
        # LaTeX label list: ordered list of all LaTeX param labels used in 
        # this waveform
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in params:
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, times, parameters):
        """
        Construct a IMR waveform.

        Parameters
        ----------
        time : array_like
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Inspect the class 
            .labels to see the required keys.

        Returns
        -------
        ndarray
            The complex waveform.
        """
        # Get the parameters from the dictionary
        mass1 = parameters[self.labels['mass1'][0]]
        mass2 = parameters[self.labels['mass2'][0]]
        spin1 = parameters[self.labels['spin1'][0]]
        spin2 = parameters[self.labels['spin2'][0]]
        distance = parameters[self.labels['distance'][0]]
        imr_inclination = parameters[self.labels['imr_inclination'][0]]
        imr_phase = parameters[self.labels['imr_phase'][0]]
        peak_time = parameters[self.labels['peak_time'][0]]

        # Generate the waveform
        hp, hc = get_td_waveform(
            approximant=self.approximant,
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1,
            spin2z=spin2,
            delta_t=1/self.fs,
            coa_phase=imr_phase,
            inclination=imr_inclination,
            distance=distance,
            f_ref=self.f_ref,
            f_lower=self.f_low
            )
        
        imr_wf = hp.data + 1j*hc.data
        
        # We need to pad/ mask the imr waveform such that it is the same length 
        # as times, and the peak occurs at peak_time
        
        # The location of the peak in the imr waveform (which we want to
        # correspond to time peak_time)
        imr_peak_index = np.argmax(abs(imr_wf))
        
        # The location of the time nearest peak_time in the times array (which 
        # is where we want to put the peak of the waveform)
        peak_index = np.argmin(abs(times - peak_time))
        
        # The number of data points before and after the peak index in the 
        # imr_wf array
        imr_pre_peak_len = imr_peak_index
        imr_post_peak_len = len(imr_wf) - 1 - imr_peak_index
        
        # The number of data points before and after the peak index in the 
        # times array
        pre_peak_len = peak_index
        post_peak_len = len(times) - 1 - peak_index
        
        # Determine by how much we need to pad/ mask the imr array
        if pre_peak_len > imr_pre_peak_len:
            pre_pad = pre_peak_len - imr_pre_peak_len
            pre_mask = None
        else:
            pre_pad = 0
            pre_mask = imr_peak_index - pre_peak_len
        
        if post_peak_len > imr_post_peak_len:
            post_pad = post_peak_len - imr_post_peak_len
            post_mask = None
        else:
            post_pad = 0
            post_mask = imr_peak_index + post_peak_len + 1
        
        # Pad the imr_wf array with zeros
        imr_wf_padded = np.pad(imr_wf, [pre_pad,post_pad])
        
        # Mask to the same length as the time array
        h = imr_wf_padded[pre_mask:post_mask]

        # Window the beginning of the waveform
        
        # We want the roll-on time to be long enough to avoid high frequency
        # contamination. In get_window we specify the fraction of the data 
        # which has the window applied. The roll-on time is half this fraction 
        # (we remove the roll-off). 1/(0.2s) = 5Hz is sufficiently low.
        T = times[-1] - times[0]
        N = len(times)
        w = get_window(('tukey', 0.4/T), N)
        
        # Remove roll-off so that the window ramps up to 1 and remains
        w[len(w)//2:] = 1
        
        # Shift the window, such that the roll-on starts when the signal starts
        signal_start = np.where(abs(h) != 0)[0][0]
        w = np.roll(w, signal_start)
        w[:signal_start] = 0
        
        return h*w
    
    
class wavelet_ringdown_sum:
    """
    Class to construct a sum of wavelets truncated at a ringdown start time,
    with a ringdown waveform attached.
    
    Parameters
    ----------
    modes : list
        A list of (l,m,n) tuples (where l is the angular number of the mode, m
        is the azimuthal number, and n is the overtone number) to include in 
        the ringdown signal. 
        
    Nwavelets : int
        The number of wavelets to include in the sum. Note that each wavelet 
        contains a positive frequency component and a negative frequency
        component (each with a amplitude and phase) for general ellipticity.
    """
    
    def __init__(self, modes, Nwavelets):
        """
        Initialize the class.
        """
        self.modes = modes
        self.Nwavelets = Nwavelets
        
        # The number of modes in this model
        self.Nmodes = len(modes)
        
        # Initialize the wavelet sum and Kerr ringdown subclasses
        self.sub_wavelet_sum = wavelet_sum(Nwavelets)
        self.sub_ringdown = ringdown(modes)
        
        # The number of parameters in this model
        self.N_params = self.sub_wavelet_sum.N_params + self.sub_ringdown.N_params
            
        self.create_labels()
        
    def create_labels(self):
        """
        Create a dictionary (self.labels) containing all the required 
        parameters for the initalized model. Also create lists of the labels
        (self.labels_text and self.labels_latex) for posterior headers and 
        plotting.
        """
        # Label dictionary: keys are the above param types, values are a list
        # of shorter param labels
        self.labels = {
            **self.sub_wavelet_sum.labels, 
            **self.sub_ringdown.labels
            }
        
        # Label list: ordered list of all param labels used in this waveform
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
            
        # LaTeX label dictionary: same as self.labels, but values are a list
        # of LaTeX labels
        self.latex_labels = {
            **self.sub_wavelet_sum.latex_labels, 
            **self.sub_ringdown.latex_labels
            }
        
        # LaTeX label list: ordered list of all LaTeX param labels used in 
        # this waveform
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, times, parameters):
        """
        Construct the wavelet + ringdown model.

        Parameters
        ----------
        times : array_like
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Inspect the class 
            .labels to see the required keys.


        Returns
        -------
        ndarray
            The complex wavelet + ringdown sum.
        """
        # Evaluate the wavelet and ringdown models
        h_w = self.sub_wavelet_sum.waveform(times, parameters)
        h_rd = self.sub_ringdown.waveform(times, parameters)
        
        # Truncate the wavelet sum at the ringdown start time with a Heaviside
        # step function. We pass x2 = 0 because in the event the ringdown 
        # start time lies exactly on a time sample, we don't want to include
        # that time in the wavelet sum (times >= than the ringdown start time
        # are masked in the ringdown function).
        H = np.heaviside(parameters[self.labels['start_time'][0]] - times, 0)
        h_w *= H
        
        # Return the sum of the models
        return h_w + h_rd
        

class imr_ringdown_sum:
    """
    Class to construct a IMR waveform truncated at a ringdown start time, with 
    a ringdown waveform attached.
    
    Parameters
    ----------
    modes : list
        A list of (l,m,n) tuples (where l is the angular number of the mode, m
        is the azimuthal number, and n is the overtone number) to include in 
        the ringdown signal. 
        
    Nwavelets : int
        The number of wavelets to include in the sum. Note that each wavelet 
        contains a positive frequency component and a negative frequency
        component (each with a amplitude and phase) for general ellipticity.
    """
    
    def __init__(self, modes, approximant='IMRPhenomD', fs=4096., f_low=20, f_ref=20):
        """
        Initialize the class.
        """
        self.modes = modes
        self.approximant = approximant
        self.fs = fs
        self.f_low = f_low
        self.f_ref = f_ref
        
        # The number of modes in this model
        self.Nmodes = len(modes)
        
        # Initialize the wavelet sum and Kerr ringdown subclasses
        self.sub_imr = imr(approximant, fs, f_low, f_ref)
        self.sub_ringdown = ringdown(modes)
        
        # The number of parameters in this model
        self.N_params = self.sub_imr.N_params + self.sub_ringdown.N_params
            
        self.create_labels()
        
    def create_labels(self):
        """
        Create a dictionary (self.labels) containing all the required 
        parameters for the initalized model. Also create lists of the labels
        (self.labels_text and self.labels_latex) for posterior headers and 
        plotting.
        """
        # Label dictionary: keys are the above param types, values are a list
        # of shorter param labels
        self.labels = {
            **self.sub_imr.labels, 
            **self.sub_ringdown.labels
            }
        
        # Label list: ordered list of all param labels used in this waveform
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
            
        # LaTeX label dictionary: same as self.labels, but values are a list
        # of LaTeX labels
        self.latex_labels = {
            **self.sub_wavelet_sum.latex_labels, 
            **self.sub_ringdown.latex_labels
            }
        
        # LaTeX label list: ordered list of all LaTeX param labels used in 
        # this waveform
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, times, parameters):
        """
        Construct the IMR + ringdown model.

        Parameters
        ----------
        times : array_like
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Inspect the class 
            .labels to see the required keys.

        Returns
        -------
        ndarray
            The complex wavelet + ringdown sum.
        """
        # Evaluate the wavelet and ringdown models
        h_imr = self.sub_imr.waveform(times, parameters)
        h_rd = self.sub_ringdown.waveform(times, parameters)
        
        # Truncate the wavelet sum at the ringdown start time with a Heaviside
        # step function. We pass x2 = 0 because in the event the ringdown 
        # start time lies exactly on a time sample, we don't want to include
        # that time in the wavelet sum (times >= than the ringdown start time
        # are masked in the ringdown function).
        H = np.heaviside(parameters[self.labels['start_time'][0]] - times, 0)
        h_imr *= H
        
        # Return the sum of the models
        return h_imr + h_rd