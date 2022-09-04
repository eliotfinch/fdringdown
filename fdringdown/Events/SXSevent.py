import numpy as np

# Package imports
from ringdown import GWevent, SXSsim
from ringdown.utils import Msun, G, c

# Modules for the spin-weighted spherical harmonics and Wigner D matrices
import spherical

# Module for quarternions
import quaternionic

# 1D interpolation
from scipy.interpolate import interp1d

# Window functions
from scipy.signal import get_window

import pickle

#%% GW150914 data

# Create the event class
event = GWevent('GW150914')

# Detector frame final mass
Mf_det = (1+event.parameters['redshift'])*event.parameters['final_mass_source']
Mf_det_seconds = Mf_det*Msun*G/c**3

# The reference Hanford frame injection time, as used in 
# https://arxiv.org/abs/2201.00822
tpeak_inj = 1126259472.423

# Injection times relative to tpeak_inj
injection_times = [-30, -25, -20, -5, 5, 10, 15, 20, 25, 30]

# Load the LIGO PSDs used in GWTC-1, https://dcc.ligo.org/LIGO-P1900011/public
asd_dict = {}
PSD_path = 'LVC_PSDs/'
for IFO_name in event.IFO_names:
    freqs, psd = np.loadtxt(f'{PSD_path}{IFO_name}.dat').T
    # Square root to get the ASD, interpolate, and store to dictionary
    asd_dict[IFO_name] = interp1d(freqs, np.sqrt(psd), bounds_error=False)

# Get interferometers
IFOs = {IFO.name: IFO for IFO in event.IFO_list}

# Sky location
sky_params = {'ra': 1.95, 'dec':-1.27, 't_event':tpeak_inj, 'psi':0.82}

# Livingston time delay from Hanford
time_delay = IFOs['L1'].time_delay(
    IFOs['H1'].vertex, 
    sky_params['ra'], 
    sky_params['dec'], 
    sky_params['t_event'])

# Subtract mean from the data
filtered_data = {
    IFO_name:event.data[IFO_name]-np.mean(event.data[IFO_name]) for IFO_name in event.IFO_names}


#%% SXS:BBH:0305 data

sim = SXSsim(305, ellMax=2, zero_time=(2,2))

# Project signal with inclination pi
# ==================================

# Initialize an empty, complex array
signal = np.zeros_like(sim.times, dtype=complex)

# Construct a Wigner object
wigner = spherical.Wigner(sim.ellMax)

# Get the quaternion representing the rotation needed
R = quaternionic.array.from_spherical_coordinates(2.68, 0)

# And the spin-weighted spherical harmonic
Y = wigner.sYlm(-2, R)

# Compute the projected signal. This is done by summing each of the modes, 
# weighted by the spin-weighted spherical harmonics
for l in range(2, sim.ellMax+1):
    for m in range(-l, l+1):
        signal += sim.h[l,m] * Y[wigner.Yindex(l,m)]
        
# Rescale time and amplitude
# ==========================

# c1, inclination pi
# r = 410*3.086e22
# M = 72*Msun

# c2, inclination 2.68:
r = event.parameters['luminosity_distance']*3.086e22
M = (1+event.parameters['redshift']) * \
    (event.parameters['mass_1_source'] + event.parameters['mass_2_source'])*Msun

M_seconds = M*G/c**3
r_seconds = r/c

signal *= M_seconds/r_seconds
signal_time = sim.times*M_seconds

# Interpolate
# ===========

signal_interp_components = {
    'real': interp1d(signal_time, np.real(signal), bounds_error=False, fill_value=0),
    'imag': interp1d(signal_time, np.imag(signal), bounds_error=False, fill_value=0)
    }

def signal_interp(times):
    
    # Evaluate interpolant
    signal = signal_interp_components['real'](times) \
        + 1j*signal_interp_components['imag'](times)
        
    # Smooth the beginning of the waveform
    
    # We want the roll-on time to be long enough to avoid high frequency
    # contamination. In get_window we specify the fraction of the data which
    # has the window applied. The roll-on time is half this fraction (we remove
    # the roll-off). 1/(0.2s) = 5Hz is sufficiently low.
    T = times[-1] - times[0]
    N = len(times)
    w = get_window(('tukey', 0.4/T), N)
    
    # Remove roll-off so that the window ramps up to 1 and remains
    w[len(w)//2:] = 1
    
    # Shift the window, such that the roll-on starts when the signal starts
    signal_start = np.where(abs(signal) != 0)[0][0]
    w = np.roll(w, signal_start)
    w[:signal_start] = 0
    
    return signal*w

#%% Create analysis data for each injection time

analysis_data = {}
analysis_times = {}

for injection_time in injection_times:
    
    # We add the SXS waveform to the data, such that the time of peak strain
    # occurs in Hanford at this time:
    tpeak = tpeak_inj + injection_time
    
    # Get the noise sample and the analysis time
    time, noise = event.get_data_segment(
        event.time,
        filtered_data,
        start_time=tpeak-2, 
        segment_length=4*event.fs, 
        window=('tukey', 0.2))
    
    # Get the interpolated SXS signal
    signal_instance = signal_interp(time-tpeak)
    
    # Project onto the IFOs
    projected_signals = {
        IFO.name: IFO.project(signal_instance, **sky_params) for IFO in event.IFO_list}
    
    # We add the Hanford signal to the noise with no time shift applied
    H_data = noise['H1'] + projected_signals['H1']
    
    # We shift the livingston signal such that the time of peak strain occurs 
    # at:
    tpeak_L = tpeak + time_delay
    tpeak_L_index = np.argmin((time-tpeak_L)**2)
    
    # Original peak index
    tpeak_index = np.argmin((time-tpeak)**2)
    
    # Apply shift
    projected_signal_L = np.roll(
        projected_signals['L1'], tpeak_L_index-tpeak_index)
    
    L_data = noise['L1'] + projected_signal_L
    
    analysis_data[injection_time] = {'H1': H_data, 'L1': L_data}
    analysis_times[injection_time] = time
    
# Noiseless injection
# ===================

# We add the SXS waveform to the data, such that the time of peak strain
# occurs in Hanford at this time:
tpeak = tpeak_inj - 10

# Get the noise sample and the analysis time
time, noise = event.get_data_segment(
    event.time,
    filtered_data,
    start_time=tpeak-2, 
    segment_length=4*event.fs, 
    window=('tukey', 0.2))

# Get the interpolated SXS signal
signal_instance = signal_interp(time-tpeak)

# Project onto the IFOs
projected_signals = {
    IFO.name: IFO.project(signal_instance, **sky_params) for IFO in event.IFO_list}

H_data = projected_signals['H1']

# We shift the livingston signal such that the time of peak strain occurs at:
tpeak_L = tpeak + time_delay
tpeak_L_index = np.argmin((time-tpeak_L)**2)

# Original peak index
tpeak_index = np.argmin((time-tpeak)**2)

# Apply shift
projected_signal_L = np.roll(
    projected_signals['L1'], tpeak_L_index-tpeak_index)

L_data = projected_signal_L

analysis_data['noiseless'] = {'H1': H_data, 'L1': L_data}
analysis_times['noiseless'] = time


#%% Save to file

with open('analysis_data_c2.pickle', 'wb') as f:
     pickle.dump(analysis_data, f)
with open('analysis_times_c2.pickle', 'wb') as f:
     pickle.dump(analysis_times, f)