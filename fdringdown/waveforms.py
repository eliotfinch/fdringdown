import numpy as np

from pykerr import spheroidal
from pykerr.qnm import qnmomega

from .qnm import qnm
from .utils import Msun, G, c, param_order

def wavelet(time, central_times, complex_amplitudes, frequencies, damping_times):
    r"""
    The base wavelet function, which has the form
    
    .. math::
        h = h_+ - ih_\times
        = \sum_n C_n e^{-(t-t_{cn})^2/\tau_n^2} e^{-i 2\pi f_n (t-t_{cn})}
        
    where :math:`C_{\ell m n}` are complex amplitudes, :math:`t_{cn}` are the
    wavelet central times, :math:`\tau_n` are the wavelet damping times, and
    :math:`f_n` are the wavelet frequencies.

    Parameters
    ----------
    time : array_like
        The times at which the model is evalulated.
        
    central_times : array_like
        The central times for the wavelets.
        
    complex_amplitudes : array_like
        The complex amplitudes of the modes.
        
    frequencies : array_like
        The (real) frequencies of the wavelets. Note that these are not the
        angular frequencies, as they are multiplied by :math:`2\pi` in the
        function.
        
    damping_times : array_like
        The damping times (or widths) of the wavelets.

    Returns
    -------
    h : ndarray
        The plus and cross components of the wavelet waveform, expressed as a
        complex number.
    """
    # Construct the waveform, summing over each wavelet
    h = np.sum([
        complex_amplitudes[n]
        *np.exp(-((time-central_times[n])/damping_times[n])**2)
        *np.exp(-1j*2*np.pi*frequencies[n]*(time-central_times[n]))
        for n in range(len(frequencies))
        ], axis=0)
    
    return h


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
        self.N_modes = len(modes)
        
        # The number of parameters in this model (a amplitude and phase for 
        # each mode, plus a inclination, azimuthal angle, start time, mass and 
        # spin)
        self.N_params = 2*self.N_modes + 5
        
        # Class to deal with converting mass and spin to a QNM frequency
        self.qnm = qnm()
        
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
        # Label dictionary
        self.labels = {}
        
        self.labels['rd_amplitudes'] = [
            f'A_rd_{n}' for n in range(self.N_modes)]
        
        self.labels['rd_phases'] = [
            f'phi_rd_{n}' for n in range(self.N_modes)]
        
        self.labels['inclination'] = ['iota']
        self.labels['azimuth'] = ['varphi']
        self.labels['start_time'] = ['t_0']
        self.labels['mass'] = ['M_f']
        self.labels['spin'] = ['chi_f']
        
        # Label list
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
                
        # LaTeX label dictionary
        self.latex_labels = {}
        
        self.latex_labels['rd_amplitudes'] = [
            fr'$A_{n}$' for n in range(self.N_modes)]
        self.latex_labels['rd_phases'] = [
            fr'$\phi_{n}$' for n in range(self.N_modes)]
        
        self.latex_labels['inclination'] = [r'$\iota$']
        self.latex_labels['azimuth'] = [r'$\varphi$']
        self.latex_labels['start_time'] = [r'$t_0^{\mathrm{geo}}$']
        self.latex_labels['mass'] = [r'$M_f\ [M_\odot]$']
        self.latex_labels['spin'] = [r'$\chi_f$']
            
        # LaTeX label list
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, times, parameters):
        r"""
        Construct a sum of the ringdown modes, using the base ringdown 
        function.

        Parameters
        ----------
        time : array
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Call the class 
            .labels() function to see the required keys.

        Returns
        -------
        array
            The complex ringdown sum.
        """
        # Get amplitudes and phases from the parameters dict
        amplitudes = np.array(
            [parameters[n] for n in self.labels['rd_amplitudes']]
            )
            
        phases = np.array(
            [parameters[n] for n in self.labels['rd_phases']]
            )
            
        # Other required parameters
        inclination = parameters[self.labels['inclination'][0]]
        azimuth = parameters[self.labels['azimuth'][0]]
        start_time = parameters[self.labels['start_time'][0]]
        mass = parameters[self.labels['mass'][0]]*self.conversion
        spin = parameters[self.labels['spin'][0]]
        
        # Obtain the (complex) QNM frequency list
        frequencies = [qnmomega(spin, *mode)/mass for mode in self.modes]
        
        # Create an empty array to add the result to
        h = np.zeros(len(times), dtype=complex)
        
        # Mask so that we only consider times after the start time
        t_mask = times >= start_time
    
        # Shift the time so that the waveform starts at time zero, and mask 
        # times after the start time
        times = (times - start_time)[t_mask]
        
        for i, (l,m,n) in enumerate(self.modes):
            h[t_mask] += amplitudes[i]*(
                np.conjugate(spheroidal(inclination, spin, l, m, n, phi=azimuth))\
                *np.exp(-1j*(frequencies[i]*times+phases[i]))\
                +spheroidal(np.pi-inclination, spin, l, m, n, phi=azimuth)\
                *np.exp(1j*(np.conjugate(frequencies[i])*times+phases[i]))
                )
        
        # Evaluate the ringdown sum
        return h
    
    def waveform_isi(self, times, parameters):
        r"""
        Construct a sum of the ringdown modes, using the base ringdown 
        function.

        Parameters
        ----------
        time : array
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Call the class 
            .labels() function to see the required keys.

        Returns
        -------
        array
            The complex ringdown sum.
        """
        # Get amplitudes and phases from the parameters dict
        amplitudes = np.array(
            [parameters[n] for n in self.labels['rd_amplitudes']]
            )
            
        phases = np.array(
            [parameters[n] for n in self.labels['rd_phases']]
            )
            
        # Other required parameters
        inclination = parameters[self.labels['inclination'][0]]
        azimuth = parameters[self.labels['azimuth'][0]]
        start_time = parameters[self.labels['start_time'][0]]
        mass = parameters[self.labels['mass'][0]]*self.conversion
        spin = parameters[self.labels['spin'][0]]
        
        # Obtain the (complex) QNM frequency list
        frequencies = [qnmomega(spin, *mode)/mass for mode in self.modes]
        
        # Create an empty array to add the result to
        h = np.zeros(len(times), dtype=complex)
        
        # Mask so that we only consider times after the start time
        t_mask = times >= start_time
    
        # Shift the time so that the waveform starts at time zero, and mask 
        # times after the start time
        times = (times - start_time)[t_mask]
        
        for i, (l,m,n) in enumerate(self.modes):
            h[t_mask] += amplitudes[i]*(
                spheroidal(inclination, spin, l, m, n, phi=azimuth)\
                *np.exp(-1j*(frequencies[i]*times-phases[i]))\
                +np.conjugate(spheroidal(np.pi-inclination, spin, l, m, n, phi=azimuth))\
                *np.exp(1j*(np.conjugate(frequencies[i])*times-phases[i]))
                )
        
        # Evaluate the ringdown sum
        return h

class wavelet_sum_v2:
    """
    """
    
    def __init__(self, Nwavelets):
        """
        Initialize the class.
        """
        self.Nwavelets = Nwavelets
        
        self.create_labels()
            
    def create_labels(self):
        """
        Create self.labels and self.latex_labels dictionaries, that contain 
        all the required parameters for the initialised model and their labels.
        Also creates lists of the labels (self.labels_list and 
        self.latex_labels_list) ordered according to param_order in utils.
        """
        # Label dictionary
        self.labels = {}
        
        self.labels['central_times'] = [
            f'eta_{n}' for n in range(self.Nwavelets)]
        
        self.labels['w_plus_amplitudes'] = [
            f'Ap_w_{n}' for n in range(self.Nwavelets)]
        self.labels['w_minus_amplitudes'] = [
            f'Am_w_{n}' for n in range(self.Nwavelets)]
        
        self.labels['w_plus_phases'] = [
            f'phip_w_{n}' for n in range(self.Nwavelets)]
        self.labels['w_minus_phases'] = [
            f'phim_w_{n}' for n in range(self.Nwavelets)]
        
        self.labels['frequencies'] = [
            f'nu_{n}' for n in range(self.Nwavelets)]
        self.labels['damping_times'] = [
            f'tau_{n}' for n in range(self.Nwavelets)]
        
        # Label list
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
            
        # LaTeX label dictionary
        self.latex_labels = {}
        
        self.latex_labels['central_times'] = [
            fr'$\eta_{n}^\mathrm{{geo}}$' for n in range(self.Nwavelets)]
        
        self.latex_labels['w_plus_amplitudes'] = [
            fr'$\mathcal{{A}}^+_{n}$' for n in range(self.Nwavelets)]
        self.latex_labels['w_minus_amplitudes'] = [
            fr"$\mathcal{{A}}^-_{n}$" for n in range(self.Nwavelets)]
        
        self.latex_labels['w_plus_phases'] = [
            fr'$\varphi^+_{n}$' for n in range(self.Nwavelets)]
        self.latex_labels['w_minus_phases'] = [
            fr"$\varphi^-_{n}$" for n in range(self.Nwavelets)]
        
        self.latex_labels['frequencies'] = [
            fr'$\nu_{n}$' for n in range(self.Nwavelets)]
        self.latex_labels['damping_times'] = [
            fr'$\tau_{n}$' for n in range(self.Nwavelets)]
        
        # LaTeX label list
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, time, parameters):
        r"""
        Construct a sum of wavelets using the base wavelet function.

        Parameters
        ----------
        time : array
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Call the class 
            .labels() function to see the required keys.

        Returns
        -------
        array
            The complex wavelet sum.
        """
        # Get required parameters from the dictionary
        central_times = np.array(
            [parameters[n] for n in self.labels['central_times']]
            )
        
        amplitudes = np.array(
            [parameters[n] for n in self.labels['w_regular_amplitudes']] + \
            [parameters[n] for n in self.labels['w_mirror_amplitudes']]
            )
            
        phases = np.array(
            [parameters[n] for n in self.labels['w_regular_phases']] + \
            [parameters[n] for n in self.labels['w_mirror_phases']]
            )
            
        frequencies = np.array(
            [parameters[n] for n in self.labels['frequencies']]
            )
            
        damping_times = np.array(
            [parameters[n] for n in self.labels['damping_times']]
            )
        
        # Extend the central_times and damping_times lists if a general 
        # ellipticity is requested
        if self.ellipticity == 'general':
            central_times = np.hstack((central_times, central_times))
            damping_times = np.hstack((damping_times, damping_times))
        
        # The wavelet function takes complex amplitudes
        complex_amplitudes = amplitudes*np.exp(1j*phases)
        
        # Construct the full frequency list
        if self.Nregular:
            regular_frequencies = frequencies
        else:
            regular_frequencies = np.array([])
        
        if self.Nmirror:
            mirror_frequencies = -frequencies
        else:
            mirror_frequencies = np.array([])
        
        frequencies = np.hstack((regular_frequencies, mirror_frequencies))
        
        # Evaluate the wavelet sum
        return wavelet(
            time, central_times, complex_amplitudes, frequencies, 
            damping_times)
class wavelet_sum:
    """
    A waveform model consisting of a sum of wavelets. There is also an option
    to choose the waveform ellipticity to match the behaviour of the 
    kerr_ringdown class, where the regular/mirror mode parameterization gives
    three choices for the ellipticity.

    Parameters
    ----------
    Nwavelets : int
        The number of wavelets to include in the sum. If ellipticity='general',
        this is the number of "elliptical" wavelets (i.e. there are actually 
        2*Nwavelets contributing to the sum).
        
    ellipticity : int or string, optional
        There are three choices for the ellipticity, to reflect the 
        kerr_ringdown waveform behaviour:
            
            - 1 : int
                each term in the wavelet sum contains only a positive 
                frequency component
            - -1 : int
                each term in the wavelet sum contains only a negative 
                frequency component
            - 'general' : str
                each term in the wavelet sum contains a positive and negative 
                frequency component, with independant amplitudes and phases.
                
        The default is 1.
    """
    
    def __init__(self, Nwavelets, ellipticity=1):
        """
        Initialize the class.
        """
        self.Nwavelets = Nwavelets
        self.ellipticity = ellipticity
        
        if self.ellipticity in [1,-1]:
            # Each term in the wavelet sum has a central time, amplitude, 
            # phase, frequency, and damping time
            self.N_params = 5*self.Nwavelets
            
            if self.ellipticity == 1:
                self.Nregular = self.Nwavelets
                self.Nmirror = 0
            else:
                self.Nregular = 0
                self.Nmirror = self.Nwavelets
            
        elif self.ellipticity == 'general':
            # There is an additional amplitude and phase for each wavelet
            self.N_params = 7*self.Nwavelets
            
            self.Nregular = self.Nwavelets
            self.Nmirror = self.Nwavelets
            
        # It will be useful to have the actual number of wavelets in the sum,
        # which depends on whether they are elliptically polarized or not
        self.Nactual = self.Nregular + self.Nmirror
        
        self.create_labels()
            
    def create_labels(self):
        """
        Create self.labels and self.latex_labels dictionaries, that contain 
        all the required parameters for the initialised model and their labels.
        Also creates lists of the labels (self.labels_list and 
        self.latex_labels_list) ordered according to param_order in utils.
        """
        # Label dictionary
        self.labels = {}
        
        self.labels['central_times'] = [
            f'eta_{n}' for n in range(self.Nwavelets)]
        
        self.labels['w_regular_amplitudes'] = [
            f'A_w_{n}' for n in range(self.Nregular)]
        self.labels['w_mirror_amplitudes'] = [
            f'Ap_w_{n}' for n in range(self.Nmirror)]
        
        self.labels['w_regular_phases'] = [
            f'phi_w_{n}' for n in range(self.Nregular)]
        self.labels['w_mirror_phases'] = [
            f'phip_w_{n}' for n in range(self.Nmirror)]
        
        self.labels['frequencies'] = [
            f'nu_{n}' for n in range(self.Nwavelets)]
        self.labels['damping_times'] = [
            f'tau_{n}' for n in range(self.Nwavelets)]
        
        # Label list
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
            
        # LaTeX label dictionary
        self.latex_labels = {}
        
        self.latex_labels['central_times'] = [
            fr'$\eta_{n}^\mathrm{{geo}}$' for n in range(self.Nwavelets)]
        
        self.latex_labels['w_regular_amplitudes'] = [
            fr'$\mathcal{{A}}_{n}$' for n in range(self.Nregular)]
        self.latex_labels['w_mirror_amplitudes'] = [
            fr"$\mathcal{{A}}'_{n}$" for n in range(self.Nmirror)]
        
        self.latex_labels['w_regular_phases'] = [
            fr'$\varphi_{n}$' for n in range(self.Nregular)]
        self.latex_labels['w_mirror_phases'] = [
            fr"$\varphi'_{n}$" for n in range(self.Nmirror)]
        
        self.latex_labels['frequencies'] = [
            fr'$\nu_{n}$' for n in range(self.Nwavelets)]
        self.latex_labels['damping_times'] = [
            fr'$\tau_{n}$' for n in range(self.Nwavelets)]
        
        # LaTeX label list
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, time, parameters):
        r"""
        Construct a sum of wavelets using the base wavelet function.

        Parameters
        ----------
        time : array
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Call the class 
            .labels() function to see the required keys.

        Returns
        -------
        array
            The complex wavelet sum.
        """
        # Get required parameters from the dictionary
        central_times = np.array(
            [parameters[n] for n in self.labels['central_times']]
            )
        
        amplitudes = np.array(
            [parameters[n] for n in self.labels['w_regular_amplitudes']] + \
            [parameters[n] for n in self.labels['w_mirror_amplitudes']]
            )
            
        phases = np.array(
            [parameters[n] for n in self.labels['w_regular_phases']] + \
            [parameters[n] for n in self.labels['w_mirror_phases']]
            )
            
        frequencies = np.array(
            [parameters[n] for n in self.labels['frequencies']]
            )
            
        damping_times = np.array(
            [parameters[n] for n in self.labels['damping_times']]
            )
        
        # Extend the central_times and damping_times lists if a general 
        # ellipticity is requested
        if self.ellipticity == 'general':
            central_times = np.hstack((central_times, central_times))
            damping_times = np.hstack((damping_times, damping_times))
        
        # The wavelet function takes complex amplitudes
        complex_amplitudes = amplitudes*np.exp(1j*phases)
        
        # Construct the full frequency list
        if self.Nregular:
            regular_frequencies = frequencies
        else:
            regular_frequencies = np.array([])
        
        if self.Nmirror:
            mirror_frequencies = -frequencies
        else:
            mirror_frequencies = np.array([])
        
        frequencies = np.hstack((regular_frequencies, mirror_frequencies))
        
        # Evaluate the wavelet sum
        return wavelet(
            time, central_times, complex_amplitudes, frequencies, 
            damping_times)
    
    
class wavelet_ringdown_sum:
    """
    """
    
    def __init__(self, Nwavelets, regular_modes, mirror_modes=[], 
                 deviation=False, wavelet_ellipticity=1, interp=True):
        """
        Initialize the class.
        """
        self.Nwavelets = Nwavelets
        self.regular_modes = regular_modes
        self.mirror_modes = mirror_modes
        self.deviation = deviation
        self.wavelet_ellipticity = wavelet_ellipticity
        
        # The number of ringdown modes in this model
        self.Nmodes = len(regular_modes) + len(mirror_modes)
        
        # Initialize the wavelet sum and Kerr ringdown subclasses
        self.sub_wavelet_sum = wavelet_sum(
            Nwavelets, ellipticity=wavelet_ellipticity)
        self.sub_kerr_ringdown = kerr_ringdown(
            regular_modes, mirror_modes, deviation, interp)
        
        # The number of parameters in this model
        self.N_params = \
            self.sub_wavelet_sum.N_params + self.sub_kerr_ringdown.N_params
            
        self.create_labels()
        
    def create_labels(self):
        """
        Create a dictionary (self.labels) containing all the required 
        parameters for the initalized model. Also create lists of the labels
        (self.labels_text and self.labels_latex) for posterior headers and 
        plotting.
        """
        # Label dictionary
        self.labels = {
            **self.sub_wavelet_sum.labels, 
            **self.sub_kerr_ringdown.labels
            }
        
        # Label list
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
            
        # LaTex label dictionary
        self.latex_labels = {
            **self.sub_wavelet_sum.latex_labels, 
            **self.sub_kerr_ringdown.latex_labels
            }
        
        # LaTeX label list
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, time, parameters):
        r"""
        Construct the wavelet + ringdown model.

        Parameters
        ----------
        time : array
            The times at which the waveform is evaluated.
            
        parameters : dict
            Dictionary containing all required parameters. Call the class 
            .labels() function to see the required keys.

        Returns
        -------
        array
            The complex wavelet sum.
        """
        # Evaluate the wavelet and ringdown models
        h_w = self.sub_wavelet_sum.waveform(time, parameters)
        h_rd = self.sub_kerr_ringdown.waveform(time, parameters)
        
        # Truncate the wavelet sum at the ringdown start time with a Heaviside
        # step function. We pass x2 = 0 because in the event the ringdown 
        # start time lies exactly on a time sample, we don't want to include
        # that time in the wavelet sum (times >= than the ringdown start time
        # are masked in the ringdown function).
        H = np.heaviside(parameters[self.labels['start_time'][0]] - time, 0)
        h_w *= H
        
        # Return the sum of the models
        return h_w + h_rd
        