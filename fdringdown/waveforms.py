import numpy as np

from .qnm import qnm
from .utils import Msun, G, c, param_order


def ringdown(time, start_time, complex_amplitudes, frequencies):
    r"""
    The base ringdown function, which has the form
    
    .. math:: 
        h = h_+ - ih_\times
        = \sum_{\ell m n} C_{\ell m n} e^{-i \omega_{\ell m n} (t - t_0)},
             
    where :math:`C_{\ell m n}` are complex amplitudes, 
    :math:`\omega_{\ell m n} = 2\pi f_{\ell m n} - \frac{i}{\tau_{\ell m n}}` 
    are complex frequencies, and :math:`t_0` is the start time of the 
    ringdown.
    
    If start_time is after the first element of the time array, the model is 
    zero-padded before that time. 
    
    The amplitudes should be given in the same order as the frequencies they
    correspond to.

    Parameters
    ----------
    time : array_like
        The times at which the model is evalulated.
        
    start_time : float
        The time at which the model begins. Should lie within the times array.
        
    complex_amplitudes : array_like
        The complex amplitudes of the modes.
        
    frequencies : array_like
        The complex frequencies of the modes. These should be ordered in the
        same order as the amplitudes.

    Returns
    -------
    h : ndarray
        The plus and cross components of the ringdown waveform, expressed as a
        complex number.
    """
    # Create an empty array to add the result to
    h = np.zeros(len(time), dtype=complex)
    
    # Mask so that we only consider times after the start time
    t_mask = time >= start_time

    # Shift the time so that the waveform starts at time zero, and mask times
    # after the start time
    time = (time - start_time)[t_mask]
        
    # Construct the waveform, summing over each mode
    h[t_mask] = np.sum([
        complex_amplitudes[n]*np.exp(-1j*frequencies[n]*time)
        for n in range(len(frequencies))], axis=0)
        
    return h


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


class kerr_ringdown:
    """
    Class to construct a Kerr ringdown waveform (that is, a ringdown where the 
    complex frequencies are determined by a mass and spin). This class calls 
    the base waveform.ringdown function to build waveforms. 
    
    Note that in the regular/mirror mode parametrization, there are three 
    choices for the waveform ellipticity: +1, -1, and 'general'. Taking the 
    (2,2,0) mode as an example:
        
        - +1 : modes = [(2,2,0)], mirror_modes = []
        - -1 : modes = [], mirror_modes = [(2,-2,0)]
        - 'general' : modes = [(2,2,0)], mirror_modes = [(2,-2,0)].
    
    Parameters
    ----------
    regular_modes : list
        A list of (l,m,n) tuples (where l is the angular number of the mode, m
        is the azimuthal number, and n is the overtone number) to include in 
        the ringdown signal. 
        
    mirror_modes : list, optional
        A list of (l,m,n) tuples (as above), specifying which 'mirror modes'
        to use in the model. The default is [] (no mirror modes are included).
        
    interp : bool, optional
        If True, use a simple interpolation to find the requested frequency. 
        This is faster than calculating the exact value. The default is True.
    """
    
    def __init__(self, regular_modes, mirror_modes=[], deviation=False, 
                 interp=True):
        """
        Initialize the class.
        """
        self.regular_modes = regular_modes
        self.mirror_modes = mirror_modes
        self.deviation = deviation
        self.interp = interp
        
        # The number of modes in this model
        self.N_modes = len(regular_modes) + len(mirror_modes)
        
        # The number of parameters in this model (a complex amplitude for each
        # mode, plus a start time, mass and spin)
        self.N_params = 2*self.N_modes + 3
        
        if deviation:
            self.N_params += 2
        
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
        
        self.labels['rd_regular_amplitudes'] = [
            f'A_rd_{n}' for n in range(len(self.regular_modes))]
        self.labels['rd_mirror_amplitudes'] = [
            f'Ap_rd_{n}' for n in range(len(self.mirror_modes))]
        
        self.labels['rd_regular_phases'] = [
            f'phi_rd_{n}' for n in range(len(self.regular_modes))]
        self.labels['rd_mirror_phases'] = [
            f'phip_rd_{n}' for n in range(len(self.mirror_modes))]
        
        self.labels['start_time'] = ['t_0']
        self.labels['mass'] = ['M_f']
        self.labels['spin'] = ['chi_f']
        
        if self.deviation:
            self.labels['frequency_deviation'] = ['delta_f']
            self.labels['damping_time_deviation'] = ['delta_tau']
        
        # Label list
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
                
        # LaTeX label dictionary
        self.latex_labels = {}
        
        self.latex_labels['rd_regular_amplitudes'] = [
            fr'$A_{n}$' for n in range(len(self.regular_modes))]
        self.latex_labels['rd_mirror_amplitudes'] = [
            fr"$A'_{n}$" for n in range(len(self.mirror_modes))]
        
        self.latex_labels['rd_regular_phases'] = [
            fr'$\phi_{n}$' for n in range(len(self.regular_modes))]
        self.latex_labels['rd_mirror_phases'] = [
            fr"$\phi'_{n}$" for n in range(len(self.mirror_modes))]
        
        self.latex_labels['start_time'] = [r'$t_0^{\mathrm{geo}}$']
        self.latex_labels['mass'] = [r'$M_f\ [M_\odot]$']
        self.latex_labels['spin'] = [r'$\chi_f$']
        
        if self.deviation:
            self.latex_labels['frequency_deviation'] = [r'$\delta f_1$']
            self.latex_labels['damping_time_deviation'] = [r'$\delta \tau_1$']
            
        # LaTeX label list
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
    
    def waveform(self, time, parameters):
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
            [parameters[n] for n in self.labels['rd_regular_amplitudes']] + \
            [parameters[n] for n in self.labels['rd_mirror_amplitudes']]
            )
            
        phases = np.array(
            [parameters[n] for n in self.labels['rd_regular_phases']] + \
            [parameters[n] for n in self.labels['rd_mirror_phases']]
            )
            
        # Other required parameters
        start_time = parameters[self.labels['start_time'][0]]
        mass = parameters[self.labels['mass'][0]]
        spin = parameters[self.labels['spin'][0]]
        
        # The ringdown function takes complex amplitudes
        complex_amplitudes = amplitudes*np.exp(1j*phases)
        
        # Obtain the (complex) QNM frequency list
        
        # The regular (positive real part) frequencies
        regular_frequencies = self.qnm.omega_list(
            self.regular_modes, spin, mass*self.conversion, self.interp)
        
        # The mirror (negative real part) frequencies can be obtained using 
        # symmetry properties 
        mirror_frequencies = -np.conjugate(self.qnm.omega_list(
            [(l,-m,n) for l,m,n in self.mirror_modes], spin, 
            mass*self.conversion, self.interp))
        
        if self.deviation:
            delta_f = parameters[self.labels['frequency_deviation'][0]]
            delta_tau = parameters[self.labels['frequency_deviation'][0]]
            omega = mirror_frequencies[1]
            omega_adjusted = np.real(omega)*np.exp(delta_f) + \
                1j*np.imag(omega)*(1/np.exp(delta_tau))
            mirror_frequencies[1] = omega_adjusted
        
        frequencies = np.hstack((regular_frequencies, mirror_frequencies))
        
        # Evaluate the ringdown sum
        return ringdown(time, start_time, complex_amplitudes, frequencies)


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
        