import numpy as np
import warnings
import sys
import shutil

from collections import namedtuple

# Define constants (SI units)
import scipy.constants as consts
G, c = consts.G, consts.c

import astropy.constants as astro_consts
Msun = astro_consts.M_sun.value

# Dynesty requires array inputs (instead of dictionaries), and so we need to
# define a order for the parameters. Use the parameter type, not the shorter
# label.
param_order = [
    
    # Wavelets
    'central_times',
    'w_plus_amplitudes',
    'w_minus_amplitudes',
    'w_plus_phases',
    'w_minus_phases',
    'frequencies',
    'damping_times',

    # IMRPhenomD
    'mass_1',
    'mass_2',
    'spin_1',
    'spin_2',
    'distance',
    'imr_inclination',
    'imr_phase',
    'peak_time',
    
    # Ringdown
    'start_time',
    'rd_amplitudes',
    'rd_phases',
    'mass',
    'spin',
    'inclination',
    'azimuth',
    
    # Extrinsic
    'right_ascension',
    'declination',
    'event_time',
    'polarization_angle'
    
    ]

# Add any parameters with periodic boundary conditions here
periodic_params = [
    
    'w_plus_phases',
    'w_minus_phases',

    'imr_phase',
    
    'rd_phases',
    'azimuth',
    
    'right_ascension',
    'polarization_angle'
    ]

# Add any parameters with reflective boundary conditions here
reflective_params = [
    'declination'
    ]

# Short labels used in the likelihood and for posterior files. Curly braces
# are placeholders for numbers (because a given model may have multiple of
# that parameter type).
param_labels = {
    
    # Wavelets
    'central_times':'eta_{}',
    'w_plus_amplitudes':'Ap_w_{}',
    'w_minus_amplitudes':'Am_w_{}',
    'w_plus_phases':'phip_w_{}',
    'w_minus_phases':'phim_w_{}',
    'frequencies':'nu_{}',
    'damping_times':'tau_{}',

    # IMRPhenomD
    'mass_1':'m_1',
    'mass_2':'m_2',
    'spin_1':'chi_1',
    'spin_2':'chi_2',
    'distance':'d_L',
    'imr_inclination':'iota_imr',
    'imr_phase':'phi_imr',
    'peak_time':'t_peak',
    
    # Ringdown
    'start_time':'t_0',
    'rd_amplitudes':'A_rd_{}',
    'rd_phases':'phi_rd_{}',
    'mass':'M_f',
    'spin':'chi_f',
    'inclination':'iota',
    'azimuth':'varphi',
    
    # Extrinsic
    'right_ascension':'ra',
    'declination':'dec',
    'event_time':'t_event',
    'polarization_angle':'psi'
    
    }

# LaTeX labels for plots
latex_labels = {
    
    # Wavelets
    'central_times':r'$\eta_{}^\mathrm{{geo}}$',
    'w_plus_amplitudes':r'$\mathcal{{A}}^+_{}$',
    'w_minus_amplitudes':r'$\mathcal{{A}}^-_{}$',
    'w_plus_phases':r'$\Phi^+_{}$',
    'w_minus_phases':r'$\Phi^-_{}$',
    'frequencies':r'$\nu_{}$',
    'damping_times':r'$\tau_{}$',

    # IMRPhenomD
    'mass_1':r'$m_1\ [M_\odot]$',
    'mass_2':r'$m_2\ [M_\odot]$',
    'spin_1':r'$\chi_1$',
    'spin_2':r'$\chi_2$',
    'distance':r'$d_L\ [\mathrm{Mpc}]$',
    'imr_inclination':r'$\iota_\mathrm{IMR}$',
    'imr_phase':r'$\phi_\mathrm{IMR}$',
    'peak_time':r'$t_\mathrm{peak}$',
    
    # Ringdown
    'start_time':r'$t_0^{\mathrm{geo}}$',
    'rd_amplitudes':r'$A_{}$',
    'rd_phases':r'$\phi_{}$',
    'mass':r'$M_f\ [M_\odot]$',
    'spin':r'$\chi_f$',
    'inclination':r'$\iota$',
    'azimuth':r'$\varphi$',
    
    # Extrinsic
    'right_ascension':r'$\alpha$',
    'declination':r'$\delta$',
    'event_time':r'$t_\mathrm{event}$',
    'polarization_angle':r'$\psi$'
    
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


def print_fn(results,
             niter,
             ncall,
             add_live_it=None,
             dlogz=None,
             stop_val=None,
             nbatch=None,
             logl_min=-np.inf,
             logl_max=np.inf):
    """
    A modified version of the print function available in dynesty (see
    https://github.com/joshspeagle/dynesty/blob/master/py/dynesty/utils.py)
    which includes newline characters. This is useful when saving output to a
    file.
    
    Parameters
    ----------
    results : tuple
        Collection of variables output from the current state of the sampler.
        Currently includes:
        (1) particle index,
        (2) unit cube position,
        (3) parameter position,
        (4) ln(likelihood),
        (5) ln(volume),
        (6) ln(weight),
        (7) ln(evidence),
        (8) Var[ln(evidence)],
        (9) information,
        (10) number of (current) function calls,
        (11) iteration when the point was originally proposed,
        (12) index of the bounding object originally proposed from,
        (13) index of the bounding object active at a given iteration,
        (14) cumulative efficiency, and
        (15) estimated remaining ln(evidence).
        
    niter : int
        The current iteration of the sampler.
        
    ncall : int
        The total number of function calls at the current iteration.
        
    add_live_it : int, optional
        If the last set of live points are being added explicitly, this
        quantity tracks the sorted index of the current live point being added.
        
    dlogz : float, optional
        The evidence stopping criterion. If not provided, the provided
        stopping value will be used instead.
        
    stop_val : float, optional
        The current stopping criterion (for dynamic nested sampling). Used if
        the `dlogz` value is not specified.
        
    nbatch : int, optional
        The current batch (for dynamic nested sampling).
        
    logl_min : float, optional
        The minimum log-likelihood used when starting sampling. Default is
        `-np.inf`.
        
    logl_max : float, optional
        The maximum log-likelihood used when stopping sampling. Default is
        `np.inf`.
    """

    fn_args = get_print_fn_args(results,
                                niter,
                                ncall,
                                add_live_it=add_live_it,
                                dlogz=dlogz,
                                stop_val=stop_val,
                                nbatch=nbatch,
                                logl_min=logl_min,
                                logl_max=logl_max)
    niter, short_str, mid_str, long_str = (fn_args.niter, fn_args.short_str,
                                           fn_args.mid_str, fn_args.long_str)

    long_str = ["iter: {:d}".format(niter)] + long_str

    # Printing.
    long_str = ' | '.join(long_str)
    mid_str = ' | '.join(mid_str)
    short_str = '|'.join(short_str)
    if sys.stderr.isatty() and hasattr(shutil, 'get_terminal_size'):
        columns = shutil.get_terminal_size(fallback=(80, 25))[0]
    else:
        columns = 200
    if columns > len(long_str):
        sys.stderr.write("\r" + long_str + ' ' * (columns - len(long_str) - 2) + "\n")
    elif columns > len(mid_str):
        sys.stderr.write("\r" + mid_str + ' ' * (columns - len(mid_str) - 2) + "\n")
    else:
        sys.stderr.write("\r" + short_str + ' ' * (columns - len(short_str) - 2) + "\n")
    sys.stderr.flush()
    

PrintFnArgs = namedtuple('PrintFnArgs',
                         ['niter', 'short_str', 'mid_str', 'long_str'])


def get_print_fn_args(results,
                      niter,
                      ncall,
                      add_live_it=None,
                      dlogz=None,
                      stop_val=None,
                      nbatch=None,
                      logl_min=-np.inf,
                      logl_max=np.inf):
    # Extract results at the current iteration.
    loglstar = results.loglstar
    logz = results.logz
    logzvar = results.logzvar
    delta_logz = results.delta_logz
    bounditer = results.bounditer
    nc = results.nc
    eff = results.eff

    # Adjusting outputs for printing.
    if delta_logz > 1e6:
        delta_logz = np.inf
    if logzvar >= 0. and logzvar <= 1e6:
        logzerr = np.sqrt(logzvar)
    else:
        logzerr = np.nan
    if logz <= -1e6:
        logz = -np.inf
    if loglstar <= -1e6:
        loglstar = -np.inf

    # Constructing output.
    long_str = []
    # long_str.append("iter: {:d}".format(niter))
    if add_live_it is not None:
        long_str.append("+{:d}".format(add_live_it))
    short_str = list(long_str)
    if nbatch is not None:
        long_str.append("batch: {:d}".format(nbatch))
    long_str.append("bound: {:d}".format(bounditer))
    long_str.append("nc: {:d}".format(nc))
    long_str.append("ncall: {:d}".format(ncall))
    long_str.append("eff(%): {:6.3f}".format(eff))
    short_str.append(long_str[-1])
    long_str.append("loglstar: {:6.3f} < {:6.3f} < {:6.3f}".format(
        logl_min, loglstar, logl_max))
    short_str.append("logl*: {:6.1f}<{:6.1f}<{:6.1f}".format(
        logl_min, loglstar, logl_max))
    long_str.append("logz: {:6.3f} +/- {:6.3f}".format(logz, logzerr))
    short_str.append("logz: {:6.1f}+/-{:.1f}".format(logz, logzerr))
    mid_str = list(short_str)
    if dlogz is not None:
        long_str.append("dlogz: {:6.3f} > {:6.3f}".format(delta_logz, dlogz))
        mid_str.append("dlogz: {:6.1f}>{:6.1f}".format(delta_logz, dlogz))
    else:
        long_str.append("stop: {:6.3f}".format(stop_val))
        mid_str.append("stop: {:6.3f}".format(stop_val))

    return PrintFnArgs(niter=niter,
                       short_str=short_str,
                       mid_str=mid_str,
                       long_str=long_str)