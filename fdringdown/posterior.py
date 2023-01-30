import numpy as np
import matplotlib.pyplot as plt

import corner

from .utils import whiten
from .waveforms import wavelet_sum
from .waveforms import ringdown

class posterior:
    
    def __init__(self, posterior_df, likelihood):
        
        self.posterior_df = posterior_df
        self.likelihood = likelihood
        
        # An array copy of the posterior will be useful for corner plots
        self.posterior_array = self.posterior_df.to_numpy(copy=True)
        
        # It will be useful to have the posterior as a list of dictionaries 
        # for plotting waveforms
        self.posterior_dict = posterior_df.to_dict('records')
        
        # Other useful things stored in the likelihood
        self.labels = np.array(likelihood.latex_labels_list)
        self.data = likelihood.data
        self.times = likelihood.times
        self.dt = likelihood.dt
        self.model = likelihood.model
        
        
    def plot_corner(self):
        """
        Create a corner plot for all parameters in the posterior. Parameters
        that are fixed with a delta function prior are automatically removed.
        The plot is saved as 'corner.png' in the current working directory.
        """
        # Mask to remove parameters with a delta function prior
        delta_mask = np.array(self.posterior_df.nunique() != 1)
        
        # Plot
        corner.corner(
            self.posterior_array[:,delta_mask], labels=self.labels[delta_mask])
        
        plt.savefig('corner.png')
        
        
    def plot_time_corner(self, IFO_name, zero_time=0):
        """
        We usually work in an Earth center frame during the analysis, so all
        times (wavelet central times, ringdown start time) are in that frame.
        It can be more useful to see the posterior on these times in an 
        interferometer frame, with some specified zero time.

        Parameters
        ----------
        IFO_name : str
            The name of the interferometer for which we change frame to.
            
        zero_time : float, optional
            A reference time in the interferometer frame. The default is 0.
        """
        # Identify the interferometer associated with the desired frame
        for IFO in self.likelihood.IFO_list:
            if IFO.name == IFO_name:
                ref_IFO = IFO
                
        posterior_copy = self.posterior_array.copy()
        
        # Identify time parameters
        time_param_locs = []
        for param_type in self.likelihood.time_params:
            time_param_locs += self.likelihood.param_locs[param_type]
        
        # Extract the time posterior samples
        time_posterior = posterior_copy[:,time_param_locs]
        time_labels = self.labels[time_param_locs]
        
        # Transform each time sample to the desired frame by calculating the
        # time delay (with the corresponding sky posterior sample)
        for i, sample in enumerate(posterior_copy):
            
            # Extract sky location params
            ra = sample[self.likelihood.param_locs['right_ascension'][0]]
            dec = sample[self.likelihood.param_locs['declination'][0]]
            t_event = sample[self.likelihood.param_locs['event_time'][0]]
            
            time_delay = ref_IFO.time_delay([0,0,0], ra, dec, t_event)
            time_posterior[i] += time_delay - zero_time
            
        # Identify samples with a delta function prior and remove them from
        # the posterior copy
        fixed_samples = []
        for i, samples in enumerate(time_posterior.T):
            if len(set(samples)) == 1:
                fixed_samples.append(i)
                
        reduced_samples = np.delete(
            time_posterior, np.s_[fixed_samples], 1)
        
        reduced_labels = list(np.delete(
            np.array(time_labels), np.s_[fixed_samples]))
        
        # Replace geo with the new frame in the label
        for i, label in enumerate(reduced_labels):
            reduced_labels[i] = label.replace('geo', IFO_name)
            if zero_time != 0:
                reduced_labels[i] += f' - {zero_time}'
        
        if len(reduced_labels) == 1:
            fig, ax = plt.subplots()
            ax.hist(reduced_samples, label=reduced_labels, bins=50)
            ax.set_xlabel(reduced_labels[0])
        else:
            corner.corner(reduced_samples, labels=reduced_labels)
            
        plt.savefig(f'{IFO_name}_frame_time_posterior.png')
        
        
    def plot_mass_spin_corner(self, truths=None):
        
        # Get the mass and spin posterior
        mass_spin_params = []
        for i, label in enumerate(self.labels):
            if 'M_f' in label:
                mass_spin_params.append(i)
            if '\chi_f' in label:
                mass_spin_params.append(i)
                
        mass_spin_posterior = self.posterior_array[:,mass_spin_params]
        mass_spin_labels = self.labels[mass_spin_params]
        
        corner.corner(
            mass_spin_posterior, labels=mass_spin_labels, truths=truths)
        plt.savefig('mass_spin_corner.png')
        
        
    def plot_waveform(self, sample_size=500, zero_time=0, 
                      xlim=(-0.1,0.05)):
        
        # Dictionary to store the waveforms for each interferometer
        waveforms = {}
        for IFO in self.likelihood.IFO_list:
            waveforms[IFO.name] = []
            
        # Take a random sample from the posterior
        sample_rows = np.random.choice(
            self.posterior_samples.shape[0], sample_size, replace=False)
        samples = self.posterior_samples[sample_rows, :]
        
        for sample in samples:
            
            # Unpack sky location parameters
            ra, dec, t_event, psi = sample[-4:]
            
            # Evaluate the model for this posterior sample
            model_waveform = self.model.waveform(self.times, sample)
            
            for IFO in self.likelihood.IFO_list:
                
                # The detector response to the model signal
                IFO_response = IFO.response(
                    model_waveform, ra, dec, t_event, psi, self.dt)
                
                waveforms[IFO.name].append(IFO_response)
                
        fig, axs = plt.subplots(nrows=len(self.likelihood.IFO_list), sharex=True, sharey=True)
        
        plot_time = self.times - zero_time
        
        for i, IFO in enumerate(self.likelihood.IFO_list):
            
            data = self.data[IFO.name]
            
            wf_5, wf_50, wf_95 = np.percentile(
                waveforms[IFO.name], q=[5, 50, 95], axis=0)
            
            axs[i].plot(plot_time, data, color='k', alpha=0.3)
            
            axs[i].plot(plot_time, wf_50, color='C0')
            axs[i].fill_between(plot_time, wf_5, wf_95, color='C0', alpha=0.3)
            
            axs[i].text(0.03, 0.83, IFO.name, transform=axs[i].transAxes)
            
        axs[0].set_xlim(xlim)
        
        axs[-1].set_xlabel(f'$t - {zero_time}\ [s]$')
        
        fig.savefig('waveforms.png', dpi=300, bbox_inches='tight')
        
        
    def plot_whitened_waveform(self, sample_size=500, zero_time=0, 
                               xlim=(-0.1,0.05), asd_dict=None):
        
        if asd_dict is None:
            asd_dict = self.likelihood.asd_dict
        
        # Dictionary to store the whitened waveforms for each interferometer
        whitened_waveforms = {}
        for IFO in self.likelihood.IFO_list:
            whitened_waveforms[IFO.name] = []
            
        # Take a random sample from the posterior
        sample_rows = np.random.choice(
            len(self.posterior_dict), sample_size, replace=False)
        samples = np.array(self.posterior_dict)[sample_rows]
        
        for sample in samples:
            
            # Unpack sky location parameters
            ra = sample[self.likelihood.labels['right_ascension'][0]]
            dec = sample[self.likelihood.labels['declination'][0]]
            t_event = sample[self.likelihood.labels['event_time'][0]]
            psi = sample[self.likelihood.labels['polarization_angle'][0]]
            
            # Evaluate the model for this posterior sample
            model_waveform = self.model.waveform(self.times, sample)
            
            for IFO in self.likelihood.IFO_list:
                
                # The detector response to the model signal
                IFO_response = IFO.response(
                    model_waveform, ra, dec, t_event, psi, self.dt)

                # Whiten the waveform and store
                whitened_waveforms[IFO.name].append(whiten(
                    IFO_response, asd_dict[IFO.name], self.dt))
        
        # Create figure
        fig, axs = plt.subplots(
            nrows=len(self.likelihood.IFO_list), sharex=True, sharey=True)
        
        plot_time = self.times - zero_time
        
        # Plot the whitened strain data and posterior waveforms for each IFO
        for i, IFO in enumerate(self.likelihood.IFO_list):
            
            # Strain data
            whitened_data = whiten(
                self.data[IFO.name], asd_dict[IFO.name], self.dt)
            
            axs[i].plot(plot_time, whitened_data, color='k', alpha=0.3)
            
            # Posterior waveforms
            wf_5, wf_50, wf_95 = np.percentile(
                whitened_waveforms[IFO.name], q=[5, 50, 95], axis=0)
            
            axs[i].plot(plot_time, wf_50, color='C0')
            axs[i].fill_between(plot_time, wf_5, wf_95, color='C0', alpha=0.3)
            
            axs[i].text(0.03, 0.83, IFO.name, transform=axs[i].transAxes)
            
        axs[0].set_xlim(xlim)
        axs[-1].set_xlabel(f'$t - {zero_time}\ [s]$')
        
        fig.savefig('whitened_waveforms.png', dpi=300, bbox_inches='tight')
        
        
    def plot_max_likelihood_waveform(self, zero_time=0, xlim=(-0.1,0.05)):
        
        # The final posterior sample corresponds to the maximum likelihood
        sample = self.posterior_dict[-1]
        
        # Evaluate model
        waveform = self.model.waveform(self.times, sample)
        
        plot_time = self.times - zero_time
                
        fig, ax = plt.subplots(figsize=(12,4))
        
        ax.plot(plot_time, np.real(waveform), label='$h_+$')
        ax.plot(plot_time, -np.imag(waveform), '--', label='$h_\\times$')
            
        ax.set_xlim(xlim)
        ax.set_xlabel(f'$t - {zero_time}\ [s]$')
        
        ax.legend()
        
        fig.savefig(
            'maximum_likelihood_waveform_geocenter.png', 
            dpi=300, 
            bbox_inches='tight')
        
    def plot_waveform_decomposition(self, zero_time=0, xlim=(-0.1,0.05)):            
        
        # The final posterior sample corresponds to the maximum likelihood
        sample = self.posterior_dict[-1]
        
        sky_params = {}
        for param_type in ['right_ascension', 'declination', 'event_time', 'polarization_angle']:
            label = self.likelihood.labels[param_type][0]
            sky_params[label] = sample[label]
        
        # Evaluate model
        waveform = self.model.waveform(self.times, sample)
        
        plot_time = self.times - zero_time
                
        fig, ax = plt.subplots(figsize=(12,4))
        
        ax.plot(plot_time, np.real(waveform), c='C0', alpha=0.5, label='$h_+$')
        ax.plot(plot_time, -np.imag(waveform), linestyle='--', c='C0', alpha=0.5, label='$h_\\times$')
        
        # if type(self.model).__name__ == 'kerr_ringdown':
        #     Nmodes = self.model.Nmodes
        #     ringdown_list = []
        #     for i, mode in enumerate(self.model.modes):
        #         ringdown_class = kerr_ringdown(modes=list(mode))
        
        if type(self.model).__name__ == 'wavelet_ringdown_sum':
            
            single_wavelet_class = wavelet_sum(
                1, ellipticity=self.model.wavelet_ellipticity)
            
            for n in range(self.model.Nwavelets):
                
                single_wavelet_params = {}
                for param_name in sample.keys():
                    if param_name[-1] == str(n):
                        new_key = param_name.replace(str(n),'0')
                        single_wavelet_params[new_key] = sample[param_name]
                
                single_wavelet_params.update(sky_params)
                
                single_wavelet_waveform = single_wavelet_class.waveform(
                    self.times, single_wavelet_params)
                
                ax.plot(plot_time, np.real(single_wavelet_waveform), c=f'C{n+1}')
                ax.plot(plot_time, -np.imag(single_wavelet_waveform), linestyle='--', c=f'C{n+1}')
                
        elif type(self.model).__name__ == 'wavelet_sum':
            
            single_wavelet_class = wavelet_sum(
                1, ellipticity=self.model.ellipticity)
            
            for n in range(self.model.Nwavelets):
                
                single_wavelet_params = {}
                for param_name in sample.keys():
                    if param_name[-1] == str(n):
                        new_key = param_name.replace(str(n),'0')
                        single_wavelet_params[new_key] = sample[param_name]
                
                single_wavelet_params.update(sky_params)
                
                single_wavelet_waveform = single_wavelet_class.waveform(
                    self.times, single_wavelet_params)
                
                ax.plot(plot_time, np.real(single_wavelet_waveform), c=f'C{n+1}')
                ax.plot(plot_time, -np.imag(single_wavelet_waveform), linestyle='--', c=f'C{n+1}')
            
        ax.set_xlim(xlim)
        ax.set_xlabel(f'$t - {zero_time}\ [s]$')
        
        ax.legend()
        
        fig.savefig(
            'maximum_likelihood_waveform_geocenter_decomposition.png', 
            dpi=300, 
            bbox_inches='tight')