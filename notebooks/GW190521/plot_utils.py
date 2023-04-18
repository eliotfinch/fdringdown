import numpy as np

rcparams = {
    # 'axes.labelsize': 18,
    # 'axes.titlesize': 24,
    'font.size': 14,
    # 'legend.fontsize': 14,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    # 'xtick.labelsize': 18,
    # 'ytick.labelsize': 18,
    'text.usetex': True,
    # 'lines.linewidth' : 1,
}


# Colors
# ------

# Colors associated with each of the t0 Gaussian priors. These were obtained
# from the "batlow" map (https://doi.org/10.5281/zenodo.5501399).
c_t0_prior = [
    (0.005193, 0.098238, 0.349842, 1.0),
    (0.066899, 0.263188, 0.377594, 1.0),
    (0.133298, 0.375282, 0.379395, 1.0),
    (0.302379, 0.450282, 0.300122, 1.0),
    (0.511253, 0.510898, 0.193296, 1.0),
    (0.754268, 0.565033, 0.211761, 1.0),
    (0.950697, 0.616649, 0.428624, 1.0),
    (0.992771, 0.707689, 0.71238, 1.0)
    ]

# Constants
# ---------

import scipy.constants as consts
G, c = consts.G, consts.c

import astropy.constants as astro_consts
Msun = astro_consts.M_sun.value

# GW190521
Mf_det = 229.944
conversion = Mf_det*Msun*G/c**3

# Maximum likelihood merger time in Livingston, as used in Capano et al. 2021
# arXiv:2105.05238
t_ref = 1242442967.4242728

# Useful functions
# ----------------

def gaussian(x, mu=0, sigma=1):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)
