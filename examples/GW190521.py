import numpy as np
import pandas as pd

import json

# Sampler
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot

# Package imports
import fdringdown as rd
from fdringdown.utils import Msun, G, c, identify_periodic_reflective, param_labels

# 1D interpolation
from scipy.interpolate import interp1d

# import multiprocessing

# Create the event class
event = rd.GWevent('GW190521')

# Get the event GPS time
gps = event.parameters['GPS']

# Detector frame final mass
Mf_det = (1+event.parameters['redshift'])*event.parameters['final_mass_source']
Mf_det_seconds = Mf_det*Msun*G/c**3

# Time of peak strain at Hanford, as used in https://arxiv.org/abs/2010.14529
t_peak = 1242442967.42871

# Get a data segment without nans, needed for GW190521
clean_time, clean_data = event.get_data_segment(
    event.time,
    event.data,
    start_time=gps-700,
    segment_length=1400*event.fs)

# Following https://arxiv.org/abs/1409.7215 we estimate the ASD using 1024
# seconds of off-source data. We use a segment of data centered on the event 
# GPS time, where we exclude 4 seconds of data also centered on the GPS time.

# Get the off-source data
pre_off_source_time, pre_off_source_data = event.get_data_segment(
    clean_time,
    clean_data,
    start_time=gps-2-512,
    segment_length=512*event.fs)

post_off_source_time, post_off_source_data = event.get_data_segment(
    clean_time,
    clean_data,
    start_time=gps+2,
    segment_length=512*event.fs)

off_source_time = np.hstack((pre_off_source_time, post_off_source_time))
off_source_data = {}
for IFO_name in event.IFO_names:
    off_source_data[IFO_name] = np.hstack(
        (pre_off_source_data[IFO_name], post_off_source_data[IFO_name]))

# Estimate the ASD
asd_dict = event.estimate_asd(
    off_source_data, nperseg=4*event.fs, noverlap=0, window=('tukey', 0.2), fs=event.fs)

# Get the analysis data. We use 4 seconds of data centered on the GPS time.
analysis_time, analysis_data = event.get_data_segment(
    clean_time,
    clean_data,
    start_time=gps-2, 
    segment_length=4*event.fs, 
    window=('tukey', 0.2))

# Get the analysis data. We use 4 seconds of data centered on the GPS time.
analysis_time, analysis_data = event.get_data_segment(
    clean_time,
    clean_data,
    start_time=gps-2, 
    segment_length=4*event.fs, 
    window=('tukey', 0.2)
    )

#%% Model and likelihood classes

Nwavelets = 1
modes = [(2,2,0)]

model = rd.wavelet_ringdown_sum(modes, Nwavelets)

fixed_params = {
    'w_minus_amplitudes': [0],
    'w_minus_phases': [0],
    'start_time': [1242442967.4218216], # t0_geo = t0_Hanford - H1.time_delay([0,0,0], ra, dec, gps)
    'inclination': [0],
    'azimuth': [0],
    'right_ascension': [0.164],
    'declination': [-1.14],
    'polarization_angle': [2.38],
    'event_time': [gps],
    }

likelihood = rd.likelihood(
    analysis_time, 
    analysis_data, 
    model, 
    event.IFO_list, 
    asd_dict, 
    fixed_params, 
    f_range=(11,1000)
    )

#%% Prior

prior_dict = {
    
    # Wavelets
    # ========
    
    'central_times': 
        ['normal', {'args':(t_peak, 50*Mf_det_seconds)}],
        
    'w_plus_amplitudes': 
        ['uniform', {'args':(0, 1e-20)}],
        
    'w_plus_phases': 
        ['uniform_periodic', {'args':(0, 2*np.pi)}],
        
    'frequencies': 
        ['uniform', {'args':(20, 100)}], # , 'hypertriangulate':True}],
        
    'damping_times': 
        ['uniform', {'args':(4*Mf_det_seconds, 80*Mf_det_seconds)}],
        
    # Ringdown
    # ========
        
    'rd_amplitudes':
        ['uniform', {'args':(0, 1e-20)}],
        
    'rd_phases': 
        ['uniform_periodic', {'args':(0, 2*np.pi)}],
        
    'mass':
        ['uniform', {'args':(100, 500)}],
        
    'spin':
        ['uniform', {'args':(0, 0.99)}],
        
    }

prior_class = rd.prior(prior_dict, likelihood, frame='H1')

# The sampler needs the number of dimensions
ndim = likelihood.N_params
print(f'Number of dimensions = {ndim}')

# We need the indices of the periodic parameters
periodic_indices, reflective_indices = \
    identify_periodic_reflective(likelihood.param_locs_without_fixed)


#%% Sampling

sampler = dynesty.NestedSampler(
   likelihood.dynesty_log_likelihood,
   prior_class.prior_transform,
   ndim,
   nlive=4000,
   sample='rwalk',
   periodic=periodic_indices,
   # reflective=reflective_indices,
   walks=2000
   )

sampler.run_nested(
    checkpoint_file='dynesty.save',
    checkpoint_every=7200
    # print_func=print_fn
    )


#%% Get the samples, re-weight, and save
results = sampler.results
samples, weights = results.samples, results.importance_weights()
new_samples = dyfunc.resample_equal(samples, weights)

posterior_df = pd.DataFrame(new_samples, columns=likelihood.labels_list_search)
for param_type, values in fixed_params.items():
    label = param_labels[param_type]
    for i, value in enumerate(values):
        posterior_df[label.format(i)] = value*np.ones(len(posterior_df))
posterior_df = posterior_df[likelihood.labels_list]
posterior_df.to_csv('posterior_samples.dat', index=False)

# Save other useful sampler output as a json
output = {
    'logz':results.logz[-1],
    'logzerr':results.logzerr[-1],
    'samples':results.samples.tolist(),
    'logwt':results.logwt.tolist(),
    'logl':results.logl.tolist(),
    'logvol':results.logvol.tolist()
    }
with open('sampler_output.json', 'w') as file:
    json.dump(output, file, indent=2)

post = rd.posterior(posterior_df, likelihood)

post.plot_corner()
post.plot_time_corner('H1', zero_time=t_peak)
post.plot_mass_spin_corner(truths=[Mf_det, 0.69])
post.plot_whitened_waveform(zero_time=t_peak, xlim=(-0.15, 0.05))
post.plot_max_likelihood_waveform(zero_time=t_peak, xlim=(-0.15, 0.05))
post.plot_waveform_decomposition(zero_time=t_peak, xlim=(-0.1, 0.1))

fig, axes = dyplot.traceplot(results, labels=likelihood.latex_labels_list)
fig.tight_layout()
fig.savefig('trace.png')
