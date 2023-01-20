import numpy as np
import pandas as pd

import os
import pickle
import json
import time

# Sampler
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot

# Package imports
import fdringdown as rd

from fdringdown import GWevent, prior, posterior, print_fn
from ringdown.waveform import wavelet_ringdown_sum
from ringdown.likelihood import FDlikelihood
from ringdown.utils import Msun, G, c, identify_periodic_reflective, param_labels

# 1D interpolation
from scipy.interpolate import interp1d

import multiprocessing

# Create the event class
event = GWevent('GW150914')

# Get the event GPS time
gps = event.parameters['GPS']

# Detector frame final mass
Mf_det = (1+event.parameters['redshift'])*event.parameters['final_mass_source']
Mf_det_seconds = Mf_det*Msun*G/c**3

# Time of peak strain at Hanford, as used in https://arxiv.org/abs/1905.00869
t_peak = 1126259462.423

# Load the LIGO PSDs used in GWTC-1, https://dcc.ligo.org/LIGO-P1900011/public
asd_dict = {}
PSD_path = 'LVC_PSDs'
for IFO_name in event.IFO_names:
    freqs, psd = np.loadtxt(f'{PSD_path}/{IFO_name}.dat').T
    # Square root to get the ASD, interpolate, and store to dictionary
    asd_dict[IFO_name] = interp1d(freqs, np.sqrt(psd), bounds_error=False)
    
# Filter
# filtered_data = event.filter_data(event.data, 4, 20, btype='highpass')
filtered_data = {
    IFO_name:event.data[IFO_name]-np.mean(event.data[IFO_name]) for IFO_name in event.IFO_names}

# Get the analysis data. We use 4 seconds of data centered on the GPS time.
analysis_time, analysis_data = event.get_data_segment(
    event.time,
    filtered_data,
    start_time=gps-2, 
    segment_length=4*event.fs, 
    window=('tukey', 0.2))

#%% Model and likelihood classes

Nwavelets = 3
modes = [(2,2,0),(2,2,1)]

model = rd.wavelet_ringdown_sum(
    Nwavelets, 
    regular_modes, 
    mirror_modes=mirror_modes, 
    wavelet_ellipticity=-1)

fixed_params = {'event_time':[gps]}

likelihood = FDlikelihood(
    analysis_time, analysis_data, model, event.IFO_list, 
    asd_dict, fixed_params, f_range=(20,1000))

#%% Prior

prior_dict = {
    
    # Wavelet priors
    # ==============
    
    'central_times': 
        ['normal', {'args':(t_peak, 50*Mf_det_seconds)}],
        
    'w_mirror_amplitudes': 
        ['uniform', {'args':(0, 1e-20)}],
        
    'w_mirror_phases': 
        ['uniform_periodic', {'args':(0, 2*np.pi)}],
        
    'frequencies': 
        ['uniform', {'args':(20, 200), 'hypertriangulate':True}],
        
    'damping_times': 
        ['uniform', {'args':(4*Mf_det_seconds, 80*Mf_det_seconds)}],
        
    # Ringdown priors
    # ===============
        
    'rd_mirror_amplitudes':
        ['uniform', {'args':(0, 1e-19)}],
        
    'rd_mirror_phases': 
        ['uniform_periodic', {'args':(0, 2*np.pi)}],
        
    'start_time': 
        ['uniform', {'args':(t_peak-15*Mf_det_seconds, t_peak+15*Mf_det_seconds)}],
        
    'mass':
        ['uniform', {'args':(40, 100)}],
        
    'spin':
        ['uniform', {'args':(0, 0.99)}],
        
    # Sky priors
    # ==========
        
    'right_ascension': 
        ['uniform_periodic', {'args':(0, 2*np.pi)}],
        
    'declination': 
        ['cos', {'args':(-np.pi/2, np.pi/2)}],
        
    'polarization_angle':
        ['uniform_periodic', {'args':(0, np.pi)}]
    }

prior_class = prior(prior_dict, likelihood, frame='H1')

# The sampler needs the number of dimensions
ndim = likelihood.N_params
print(f'Number of dimensions = {ndim}')

# We need the indices of the periodic parameters
periodic_indices, reflective_indices = \
    identify_periodic_reflective(likelihood.param_locs_search)


#%% Sampling

if not os.path.isdir('checkpoints/'):
    os.mkdir('checkpoints/')

save_times = []

pool = multiprocessing.Pool(processes=8)

iterations_per_call = 5e7
previous_iterations = 0

checkpoint_number = 0

if os.path.isfile(f'checkpoints/checkpoint_{checkpoint_number}.pickle'):
    print('Resuming previous run...')
    with open(f'checkpoints/checkpoint_{checkpoint_number}.pickle', 'rb') as f:
        sampler = pickle.load(f)
    sampler.rstate = np.random
    sampler.pool = pool
    sampler.M = pool.map

else:
    sampler = dynesty.NestedSampler(
       likelihood.dynesty_log_likelihood,
       prior_class.prior_transform,
       ndim,
       pool=pool,
       queue_size=8,
       nlive=4000,
       sample='rwalk',
       periodic=periodic_indices,
       reflective=reflective_indices,
       walks=2000)

finished = False

while not finished:

    sampler.run_nested(
        maxcall=iterations_per_call,
        print_func=print_fn
        )

    iterations_after_run = np.sum(sampler.results.ncall)

    with open(f'checkpoints/checkpoint_{iterations_after_run}.pickle', 'wb') as f:
         pickle.dump(sampler, f)

    save_times.append(time.asctime(time.gmtime(time.time())))

    with open('checkpoints/save_times.txt', 'w') as f:
        for entry in save_times:
            f.write(entry+'\n')

    if iterations_after_run == previous_iterations:
        finished = True

    previous_iterations = iterations_after_run

pool.close()

#%% Get the samples, re-weight, and save
results = sampler.results
samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
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

post = posterior(posterior_df, likelihood)

post.plot_corner()
post.plot_time_corner('H1', zero_time=t_peak)
post.plot_mass_spin_corner(truths=[Mf_det, 0.69])
post.plot_whitened_waveform(zero_time=t_peak, xlim=(-0.15, 0.05))
post.plot_max_likelihood_waveform(zero_time=t_peak, xlim=(-0.15, 0.05))
post.plot_waveform_decomposition(zero_time=t_peak, xlim=(-0.1, 0.1))

fig, axes = dyplot.traceplot(results, labels=likelihood.latex_labels_list)
fig.tight_layout()
fig.savefig('trace.png')

