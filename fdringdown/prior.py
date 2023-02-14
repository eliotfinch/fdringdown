import numpy as np

from scipy.special import ndtri

from .interferometer import interferometer
from .utils import hypertriangulate as hypertriangulate_func


class prior:
    
    def __init__(self, prior_dict, likelihood, frame='geo', joint_priors=[]):
        
        self.prior_dict = prior_dict
        self.likelihood = likelihood
        self.frame = frame
        self.joint_priors = joint_priors
        self.joint_priors_list = [
            param_type for pair in joint_priors for param_type in pair]
        
        if frame != 'geo':
            self.IFO = interferometer(frame)
        
        self.prior_funcs = {
            'uniform': self.uniform,
            'uniform_sum': self.uniform_sum,
            'uniform_periodic': self.uniform_periodic,
            'normal': self.normal,
            'cos': self.cos,
            'sin': self.sin
            }
        
    def td(self, ra, dec, t_event):
        return self.IFO.time_delay([0,0,0], ra, dec, t_event)
        
    def uniform(self, params, args, hypertriangulate=False):
        
        if hypertriangulate:
            params = hypertriangulate_func(params)
            
        lower, upper = args
        return params*(upper-lower) + lower
    
    def uniform_periodic(self, params, args):
        
        lower, upper = args
        return (params%1)*(upper-lower) + lower
    
    def uniform_sum(self, params, args):
        # See `amplitude_prior_transform` in examples for more information
        
        x, y = params
        lower, upper = args
        
        gamma = lower*(upper/lower)**x
        delta = gamma*(2*y - 1)
        
        alpha = (gamma + delta)/2
        beta = (gamma - delta)/2
        
        return [alpha, beta]
    
    def normal(self, params, args, hypertriangulate=False):
        
        if hypertriangulate:
            params = hypertriangulate_func(params)
            
        mu, sigma = args
        return mu + sigma*ndtri(params)
    
    def cos(self, params, args, hypertriangulate=False):
        
        if hypertriangulate:
            params = hypertriangulate_func(params)
            
        lower, upper = args
        return np.arcsin(params*(np.sin(upper)-np.sin(lower)) + np.sin(lower))
    
    def sin(self, params, args):
        
        lower, upper = args
        norm = 1/(np.cos(lower) - np.cos(upper))
        return np.arccos(np.cos(lower) - params/norm)
        
    def prior_transform(self, params):
        
        for param_type, i in self.likelihood.param_locs_without_fixed.items():
            if param_type not in self.joint_priors_list:
                prior_type, prior_params = self.prior_dict[param_type]
                params[i] = self.prior_funcs[prior_type](params[i], **prior_params)
                
        for param_type_i, param_type_j in self.joint_priors:
            i = self.likelihood.param_locs_without_fixed[param_type_i]
            j = self.likelihood.param_locs_without_fixed[param_type_j]
            prior_type, prior_params = self.prior_dict[param_type_i]
            params[i], params[j] = \
                self.prior_funcs[prior_type]([params[i],params[j]], **prior_params)
        
        if self.frame != 'geo':
            
            if 'right_ascension' in self.likelihood.fixed_params:
                ra = self.likelihood.fixed_params['right_ascension'][0]
            else:
                ra = params[self.likelihood.param_locs_without_fixed['right_ascension']][0]
                
            if 'declination' in self.likelihood.fixed_params:
                dec = self.likelihood.fixed_params['declination'][0]
            else:
                dec = params[self.likelihood.param_locs_without_fixed['declination']][0]
                
            if 'event_time' in self.likelihood.fixed_params:
                t_event = self.likelihood.fixed_params['event_time'][0]
            else:
                t_event = params[self.likelihood.param_locs_without_fixed['event_time']][0]
                    
            for param_type in self.likelihood.time_params:
                param_loc = self.likelihood.param_locs_without_fixed[param_type]
                params[param_loc] -= self.td(ra, dec, t_event)
        
        return params
        