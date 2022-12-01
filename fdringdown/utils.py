import numpy as np
import warnings

# Define constants (SI units)
Msun = 1.9884e30
G = 6.674e-11
c = 299792458.0

# Dynesty requires array inputs (instead of dictionaries), and so we need to
# define a order for the parameters. Use the parameter type, not the shorter
# label.
# param_order = [
#     'central_times',
#     'w_regular_amplitudes',
#     'w_mirror_amplitudes',
#     'w_regular_phases',
#     'w_mirror_phases',
#     'frequencies',
#     'damping_times',
#     'rd_regular_amplitudes',
#     'rd_mirror_amplitudes',
#     'rd_regular_phases',
#     'rd_mirror_phases',
#     'start_time',
#     'mass',
#     'spin',
#     'frequency_deviation',
#     'damping_time_deviation',
#     'right_ascension',
#     'declination',
#     'event_time',
#     'polarization_angle'
#     ]

param_order = [
    'central_times',
    'w_plus_amplitudes',
    'w_minus_amplitudes',
    'w_plus_phases',
    'w_minus_phases',
    'frequencies',
    'damping_times',
    'rd_amplitudes',
    'rd_phases',
    'inclination',
    'azimuth',
    'start_time',
    'mass',
    'spin',
    'right_ascension',
    'declination',
    'event_time',
    'polarization_angle'
    ]

# Add any parameters with periodic boundary conditions here
periodic_params = [
    'w_plus_phases',
    'w_minus_phases',
    'rd_phases',
    'right_ascension',
    'polarization_angle'
    ]

# Add any parameters with reflective boundary conditions here
reflective_params = [
    'declination'
    ]

param_labels = {
    'central_times':'eta_{}',
    'w_plus_amplitudes':'Ap_w_{}',
    'w_minus_amplitudes':'Am_w_{}',
    'w_plus_phases':'phip_w_{}',
    'w_minus_phases':'phim_w_{}',
    'frequencies':'nu_{}',
    'damping_times':'tau_{}',
    'rd_amplitudes':'A_rd_{}',
    'rd_phases':'phi_rd_{}',
    'inclination':'iota',
    'azimuth':'varphi',
    'start_time':'t_0',
    'mass':'M_f',
    'spin':'chi_f',
    'right_ascension':'ra',
    'declination':'dec',
    'event_time':'t_event',
    'polarization_angle':'psi'
    }


def whiten(data, asd_func, dt, f_range=(32,512)):
    """
    Whiten some data according to a provided ASD, normalising to be in units
    of the standard deviation of the noise.

    Parameters
    ----------
    data : array
        The data to whiten.

    asd_func : func
        An interpolated amplitude spectrum.

    dt : float
        The time resoltuion of the data.

    frange : tuple, optional
        Outside this frequency range we fill the ASD with infs, which upon
        division of the data by the ASD effectively bandpasses the data. The
        default is (32,512).

    RETURNS
    -------
    whitened data : array
        The whitened, normalized data.
    """
    # Number of data points
    N = len(data)

    # Calculate possible frequency bins
    freqs = np.fft.rfftfreq(N, dt)

    # FFT to frequency domain. We don't worry about correct factors because we
    # inverse FFT back to time domain later.
    data_f = np.fft.rfft(data)

    # Evaluate ASD at frequencies of interest
    asd = asd_func(freqs)

    # Replace frequencies bins outside of the mask range with infs
    mask = (freqs>=f_range[0]) & (freqs<f_range[1])
    asd[~mask] = np.inf

    # Divide by ASD
    data_f_w = data_f/asd

    # FFT back to time domain
    data_w = np.fft.irfft(data_f_w, n=N)

    # Normalize to be in units of standard deviation of the noise
    normalization = np.sqrt(2*dt)

    return data_w * normalization


def hypertriangulate(x, bounds=(0,1)):
    """
    Transform a vector of numbers from a cube to a hypertriangle.
    The hypercube is the space the samplers work in, and the hypertriangle is
    the physical space where the components of x are ordered such that
    :math:`x_0 < x_1 < \ldots < x_n`. The (unit) transformation is defined by:

    .. math::
        X_j = 1 - \\prod_{i=0}^{j} (1 - x_i)^{1/(K-i)}.

    See https://arxiv.org/abs/1907.11631 for details.

    Parameters
    ----------
    x : array
        The hypercube parameter values.

    bounds : tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle.

    Returns
    -------
    X : array
        The hypertriangle parameter values.
    """

    # Transform to the unit hypercube
    unit_x = (np.array(x) - bounds[0]) / (bounds[1] - bounds[0])

    # Hypertriangle transformation
    with warnings.catch_warnings():
        # This specific warning is raised when unit_x goes outside [0, 1]
        warnings.filterwarnings('error', 'invalid value encountered in power')
        try:
            K = np.size(unit_x)
            index = np.arange(K)
            inner_term = np.power(1 - unit_x, 1/(K - index))
            unit_X = 1 - np.cumprod(inner_term)
        except RuntimeWarning:
            raise ValueError('Values outside bounds passed to hypertriangulate')

    # Re-apply orginal scaling, offset
    X = bounds[0] + unit_X * (bounds[1] - bounds[0])

    return X


def remove_empty_keys(input_dict):
    """
    For an input dictionary where the value for each key is a single list, this
    function removes any entries where the list is empty.

    Parameters
    ----------
    input_dict : dict
        A dictionary where each entry is a single list.

    Returns
    -------
    output_dict : dict
        A copy of input_dict with empty-list entries removed.
    """
    output_dict = input_dict.copy()

    for key, value in input_dict.items():
        if len(value) == 0:
            output_dict.pop(key)

    return output_dict


def identify_periodic_reflective(param_locs):
    """
    For a dictionary with keys corresponding to parameter types and values to
    lists of the positions of those parameters, return a list containing the
    positions of periodic parameters and a list contaning positions of
    reflective parameters.

    Parameters
    ----------
    param_locs : dict
        DESCRIPTION.

    Returns
    -------
    periodic_list, reflective_list : lists
        DESCRIPTION.
    """
    periodic_list = []
    reflective_list = []

    for param_type, locs in param_locs.items():
        if param_type in periodic_params:
            periodic_list += locs
        if param_type in reflective_params:
            reflective_list += locs

    return periodic_list, reflective_list

