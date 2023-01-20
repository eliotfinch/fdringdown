import numpy as np

from .utils import param_order, param_labels, latex_labels, remove_empty_keys


class likelihood:
    """
    A class to hold a frequency-domain log-likelihood function. See the 
    form_of_the_likelihood notebook in examples for more details. This class
    also contains methods to view lists of parameter labels, and a dictionary 
    of parameter names.

    Parameters
    ----------
    times : array
        The times at which the waveform is evaluated.
        
    data : dict
        The data to use in the likelihood calculation. This should be a 
        dictionary with keys corresponding to the interferometer names.
        
    model : class
        The model class to use in the likelihood calculation. See available
        classes in fdringdown.waveforms.
        
    IFO_list : list
        A list of interferometer objects to include in the analysis. This is 
        stored as .IFO_list in the GWevent and NRevent classes.
        
    asd_dict : dict
        The ASD of the noise. This should be a dictionary with keys 
        corresponding to the interferometer names.
        
    fixed_params : dict, optional
        A dictionary with keys corresponding to the parameter type (e.g. 
        rd_amplitudes), and values corresponding to a list of fixed parameter
        values. Currently all parameters of a particular type have to be fixed
        (i.e. there is no way to fix only one ringdown amplitude and leave
        another free).
        
    f_range : tuple, optional
        The lower and upper frequencies of the likelihood integral in Hz. The 
        default is (32,512).
    """
    
    def __init__(self, times, data, model, IFO_list, asd_dict, fixed_params={},
                 f_range=(32,512)):
        """
        Initialize the class.
        """
        self.times = times
        self.data = data
        self.model = model
        self.IFO_list = IFO_list
        self.asd_dict = asd_dict
        self.fixed_params = fixed_params
        self.f_range = f_range
        
        # Time resolution
        self.dt = times[1] - times[0]

        # Number of data points in the analysis sample
        K = len(times)
        
        # Time duration
        self.T = K*self.dt

        # FFT frequencies
        self.freqs = np.fft.rfftfreq(K, d=self.dt)
        
        # Mask for the frequency range
        self.mask = (self.freqs>=f_range[0]) & (self.freqs<f_range[-1])
        
        # Convert each asd function in the provided dictionary into a psd
        # array, masked by the requested frequency range
        self.psd = {}
        for IFO in IFO_list:
            self.psd[IFO.name] = asd_dict[IFO.name](self.freqs)[self.mask]**2
            
        self.create_labels()
                
        # A dictionary with the location of each parameter in the labels_list 
        # and labels_list_without_fixed will be useful
        self.param_locs = {}
        for param_type, labels in self.labels.items():
            self.param_locs[param_type] = []
            for label in labels:
                self.param_locs[param_type].append(self.labels_list.index(label))
        self.param_locs = remove_empty_keys(self.param_locs)
        
        self.param_locs_without_fixed = {}
        for param_type, labels in self.labels_without_fixed.items():
            self.param_locs_without_fixed[param_type] = []
            for label in labels:
                self.param_locs_without_fixed[param_type].append(
                    self.labels_list_without_fixed.index(label)
                    )
        self.param_locs_without_fixed = remove_empty_keys(self.param_locs_without_fixed)
        
        # Keep track of parameters that change depending on frame
        self.time_params = []
        for param_type in ['central_times', 'start_time']:
            if param_type in self.param_locs_without_fixed.keys():
                self.time_params.append(param_type)
                
        # Create a version of fixed_params where the keys are the parameter
        # labels, not the parameter type
        self.fixed_params_merge = {}
        for param_type, values in fixed_params.items():
            label = param_labels[param_type]
            if len(values) > 1:
                for i, value in enumerate(values):
                    self.fixed_params_merge[label.format(i)] = value
            else:
                self.fixed_params_merge[label] = values[0]
        
        # The new number of parameters, with extrinsic parameters included and
        # fixed parameters removed
        self.N_params = model.N_params + 4 - len(self.fixed_params_merge)
        
    def create_labels(self):
        """
        Create a dictionary (self.labels) containing all the required 
        parameters for the initalized likelihood (this differs from the model 
        labels because we now include extrinsic parameters). Also create lists
        of the labels (self.labels_list and self.latex_labels_list) for 
        posterior files and plotting.
        """
        # The additional extrinsic parameter "types" used in this waveform 
        # (this should be a subset of the param_order list from utils)
        extrinsic_params = [
            'right_ascension',
            'declination',
            'event_time',
            'polarization_angle'
            ]
        
        extrinsic_labels = {}
        for param_type in extrinsic_params:
            extrinsic_labels[param_type] = [param_labels[param_type]]
            
        extrinsic_labels_latex = {}
        for param_type in extrinsic_params:
            extrinsic_labels_latex[param_type] = [latex_labels[param_type]]
        
        # Label dictionary: keys are the param types (extrinsic and model), 
        # values are a list of shorter param labels
        self.labels = {
            **self.model.labels,
            **extrinsic_labels
            }
        
        # Make a copy of the labels dict which doesn't include the fixed 
        # parameters
        self.labels_without_fixed = self.labels.copy()
        for param_type in self.fixed_params:
            if param_type in self.labels_without_fixed:
                self.labels_without_fixed.pop(param_type)
        
        # Label list: ordered list of all param labels used in this likelihood
        self.labels_list = []
        for param_type in param_order:
            if param_type in self.labels.keys():
                self.labels_list += self.labels[param_type]
                
        self.labels_list_without_fixed = []
        for param_type in param_order:
            if param_type in self.labels_without_fixed.keys():
                self.labels_list_without_fixed += self.labels_without_fixed[param_type]
        
        # LaTeX label dictionary: same as self.labels, but values are a list
        # of LaTeX labels
        self.latex_labels = {
            **self.model.latex_labels,
            **extrinsic_labels_latex
            }
        
        # LaTeX label list: ordered list of all LaTeX param labels used in 
        # this likelihood
        self.latex_labels_list = []
        for param_type in param_order:
            if param_type in self.latex_labels.keys():
                self.latex_labels_list += self.latex_labels[param_type]
            
    def log_likelihood(self, parameters):
        """
        Calculate the log-likelihood.

        Parameters
        ----------
        parameters : dict
            A dictionary containing all required parameters. The items of the
            class .labels_without_fixed are the required keys.

        Returns
        -------
        log_likelihood : float
            The value of the log-likelihood for the provided parameters.
        """
        # Merge with fixed parameters
        parameters = {**parameters, **self.fixed_params_merge}
        
        # Unpack sky location parameters
        ra = parameters[self.labels['right_ascension'][0]]
        dec = parameters[self.labels['declination'][0]]
        t_event = parameters[self.labels['event_time'][0]]
        psi = parameters[self.labels['polarization_angle'][0]]
        
        # We will sum the log likelihoods for each interferometer
        log_likelihood = 0

        # Create an instance of the model
        model_waveform = self.model.waveform(self.times, parameters)

        # Calculate the likelihood for each requested interferometer
        for IFO in self.IFO_list:
            
            # The detector response to the model signal
            IFO_response = IFO.response(
                model_waveform, ra, dec, t_event, psi, self.dt)

            # Calculate the difference between data and model
            diff = self.data[IFO.name] - IFO_response
            
            # FFT into frequency domain and mask
            diff_f = self.dt*np.fft.rfft(diff)[self.mask]
            
            # Perform likelihood calculation
            log_likelihood += np.sum(
                -2*abs(diff_f)**2/(self.T*self.psd[IFO.name])
                - np.log(0.5*np.pi*self.T*self.psd[IFO.name]))

        return log_likelihood
    
    def dynesty_log_likelihood(self, parameters):
        """
        A wrapper for the log-likelihood function that accepts a list of 
        parameters instead of a dictionary. 

        Parameters
        ----------
        parameters : list
            A list of all the parameter values. The order of the list should 
            follow the convention defined by param_order in utils.

        Returns
        -------
        log_likelihood : float
            The value of the log-likelihood for the provided parameters.
        """
        # Convert the list to a dictionary with the correct keys
        parameters_dict = {
            self.labels_list_search[i]: param for i, param in enumerate(parameters)
            }
        
        return self.log_likelihood(parameters_dict)
    