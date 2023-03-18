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
        
        # If the user has specified the prior via the parameter type, then we 
        # can apply the transform in parallel (this allows hypertriangulation,
        # for example)
        grouped_transform = {}
        
        # If the user has specified the prior via the parameter label, then
        # we apply the prior transform on this parameter individually
        single_transform = {}
        
        # Finally, the user might have a prior on a non-standard parameter
        # (such as an amplitude ratio), which needs to be dealt with
        self.special_transform = {}
        
        for key, value in self.prior_dict.items():
            
            if key in self.likelihood.labels_without_fixed:
                grouped_transform[key] = self.likelihood.param_locs_without_fixed[key]
                
            elif key in self.likelihood.labels_list_without_fixed:
                single_transform[key] = self.likelihood.labels_list_without_fixed.index(key)
                
            else:
                # There will be a better way to do this, but for amplitude 
                # ratios we know the parameter label we're after is given by
                target_key = key[:-7]
                # (this cuts the '_over_n' part of the string)
                
                self.special_transform[key] = self.likelihood.labels_list_without_fixed.index(target_key)
        
        # The single and grouped transforms behave the same in the prior 
        # transform, so we merge them
        self.param_locs = {**single_transform, **grouped_transform}
        
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
        # See `amplitude_prior_transform` in notebooks for more information
        
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
        
        for param_type, i in self.param_locs.items():
            # if param_type not in self.joint_priors_list:
            prior_type, prior_params = self.prior_dict[param_type]
            params[i] = self.prior_funcs[prior_type](params[i], **prior_params)
            
        for param, i in self.special_transform.items():
            
            # Apply the prior transform
            prior_type, prior_params = self.prior_dict[param]
            transformed_param = self.prior_funcs[prior_type](params[i], **prior_params)
            
            # Convert this to the parameter that the likelihood wants
            params[i] = transformed_param*params[self.likelihood.labels_list_without_fixed.index('A_rd_0')]
            
        # for param_type_i, param_type_j in self.joint_priors:
        #     i = self.likelihood.param_locs_without_fixed[param_type_i]
        #     j = self.likelihood.param_locs_without_fixed[param_type_j]
        #     prior_type, prior_params = self.prior_dict[param_type_i]
        #     params[i], params[j] = \
        #         self.prior_funcs[prior_type]([params[i],params[j]], **prior_params)
        
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
        