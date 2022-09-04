import numpy as np
import qnm as qnm_loader

from scipy.interpolate import interp1d


class qnm:
    """
    Class for loading quasinormal mode (QNM) frequencies and spherical-
    spheroidal mixing coefficients. This makes use of the qnm package,
    https://arxiv.org/abs/1908.10377
    """
    
    def __init__(self):
        """
        Initialise the class.
        """
        
        # Dictionary to store the qnm functions
        self.qnm_funcs = {}
        
        # Dictionary to store interpolated qnm functions for quicker 
        # evaluation
        self.interpolated_qnm_funcs = {}
        
    def interpolate(self, l, m, n):
        
        qnm_func = self.qnm_funcs[l,m,n]
        
        # Extract relevant quantities
        spins = qnm_func.a
        real_omega = np.real(qnm_func.omega)
        imag_omega = np.imag(qnm_func.omega)
        all_real_mu = np.real(qnm_func.C)
        all_imag_mu = np.imag(qnm_func.C)

        # Interpolate omegas
        real_omega_interp = interp1d(
            spins, real_omega, kind='cubic', bounds_error=False, 
            fill_value=(real_omega[0],real_omega[-1]))
        
        imag_omega_interp = interp1d(
            spins, imag_omega, kind='cubic', bounds_error=False, 
            fill_value=(imag_omega[0],imag_omega[-1]))
        
        # Interpolate mus
        mu_interp = []
        
        for real_mu, imag_mu in zip(all_real_mu.T, all_imag_mu.T):
            
            real_mu_interp = interp1d(
                    spins, real_mu, kind='cubic', bounds_error=False, 
                    fill_value=(real_mu[0],real_mu[-1]))
                
            imag_mu_interp = interp1d(
                spins, imag_mu, kind='cubic', bounds_error=False, 
                fill_value=(imag_mu[0],imag_mu[-1]))
            
            mu_interp.append((real_mu_interp, imag_mu_interp))

        # Add these interpolated functions to the frequency_funcs dictionary
        self.interpolated_qnm_funcs[l,m,n] = [
            (real_omega_interp, imag_omega_interp), mu_interp]
    
    # MAKE THIS p?
    # https://github.com/maxisi/ringdown/blob/main/ringdown/qnms.py#L77
    def omega(self, l, m, n, sign, chif, Mf=1, interp=False):
        """
        Return a complex frequency, :math:`\omega_{\ell m n}(M_f, \chi_f)`,
        for a particular mass, spin, and mode.
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float, optional
            The mass of the final black hole. This is the factor which the QNM
            frequencies are divided through by, and so determines the units of 
            the returned quantity. 
            
            If Mf is in units of seconds, then the returned frequency has units 
            :math:`\mathrm{s}^{-1}`. 
            
            When working with SXS simulations and GW surrogates, we work in 
            units scaled by the total mass of the binary system, M. In this 
            case, providing the dimensionless Mf value (the final mass scaled 
            by the total binary mass) will ensure the QNM frequencies are in 
            the correct units (scaled by the total binary mass). This is 
            because the frequencies loaded from file are scaled by the remnant 
            black hole mass (Mf*omega). So, by dividing by the remnant black 
            hole mass scaled by the total binary mass (Mf/M), we are left with
            Mf*omega/(Mf/M) = M*omega.
            
            The default is 1, in which case the frequencies are returned in
            units of the remnant black hole mass.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is False.
            
        Returns
        -------
        complex
            The complex QNM frequency.
        """
        # Load the correct qnm based on the type we want
        m *= sign
        
        # Test if the qnm function has been loaded for the requested mode
        if (l,m,n) not in self.qnm_funcs:
            self.qnm_funcs[l,m,n] = qnm_loader.modes_cache(-2, l, m, n)
        
        # We only cache values if not interpolating
        store = False if interp else True

        # Access the relevant functions from the qnm_funcs dictionary, and 
        # evaluate at the requested spin. Storing speeds up future evaluations.
        omega, A, mu = self.qnm_funcs[l,m,n](
            chif, store=store, interp_only=interp)
        
        # Use symmetry properties to get the mirror mode, if requested
        if sign == -1:
            omega = -np.conjugate(omega)
        
        # Return the scaled complex frequency
        return omega/Mf
    
    def omega_list(self, modes, chif, Mf=1, interp=False):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin)
        
        Parameters
        ----------            
        modes : array_like
            A sequence of (l,m,n) tuples to specify which QNMs to load 
            frequencies for. For nonlinear modes, the tuple has the form 
            (l1,m1,n1,l2,m2,n2,...).
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float, optional
            The mass of the final black hole. See the qnm.omega docstring for
            details on units. The default is 1.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is False.
            
        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the qnm function and append the result to the
        # list
        
        # Code for linear QNMs:
        # return [self.omega(l, m, n, sign, chif, Mf, interp) for l, m, n, sign in modes]
        
        # Code for nonlinear QNMs:
        return [
            sum([self.omega(l, m, n, sign, chif, Mf, interp) 
                 for l, m, n, sign in [mode[i:i+4] for i in range(0, len(mode), 4)]
                 ]) 
            for mode in modes
            ]
    
        # Writen out, the above is doing the following:
            
        # return_list = []
        # for mode in modes:
        #     sum_list = []
        #     for i in range(0, len(mode), 4):
        #         l, m, n, sign = mode[i:i+4]
        #         sum_list.append(self.omega(l, m, n, sign, chif, Mf, interp))
        #     return_list.append(sum(sum_list))
        # return return_list
    
    def omegaoft(self, l, m, n, sign, chioft, Moft=1, interp=True):
        """
        Return an array of complex frequencies corresponding to an array of
        spin and mass values. This is designed to be used with time
        dependant spin and mass values, to get a time dependant frequency
        (hence the name).
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        chioft : array_like
            The dimensionless spin magnitude of the black hole.
            
        Moft : array_like, optional
            The time dependant mass of the black hole. See the qnm.omega
            docstring for details on units. This can either be a float, which
            then divides through the whole omega array, or an array of the
            same length as chioft. The default is 1, in which case
            no scaling is performed.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is True.
            
        Returns
        -------
        ndarray
            The complex QNM frequency array, with the same length as chioft.
        """
        # Load the correct qnm based on the type we want
        m *= sign
        
        # Test if the qnm function has been loaded for the requested mode
        if (l,m,n) not in self.qnm_funcs:
            self.qnm_funcs[l,m,n] = qnm_loader.modes_cache(-2, l, m, n)
            
        # Test if the interpolated qnm function has been created
        if (l,m,n) not in self.interpolated_qnm_funcs:
            self.interpolate(l,m,n)
            
        if interp:
            
            # We create our own interpolant so that we can evaulate the 
            # frequencies for all spins simultaneously
            omega_interp = self.interpolated_qnm_funcs[l,m,n][0]
            omegaoft = omega_interp[0](chioft) + 1j*omega_interp[1](chioft)
            
            # Use symmetry properties to get the mirror mode, if requested
            if sign == -1:
                omegaoft = -np.conjugate(omegaoft)
            
            return omegaoft/Moft
        
        else:
            # List to store the frequencies corresponding to each mass and spin
            omegaoft = []
            
            for chi in chioft:
                
                # Access the relevant functions from the qnm_funcs dictionary, 
                # and evaluate at the requested spin. Storing speeds up future 
                # evaluations.
                omega, A, mu = self.qnm_funcs[l,m,n](chi, store=True)
                
                omegaoft.append(omega)
                
            # Use symmetry properties to get the mirror mode, if requested
            if sign == -1:
                omegaoft = -np.conjugate(omegaoft)
            
            # Return the scaled complex frequency
            return np.array(omegaoft)/Moft
    
    def omegaoft_list(self, modes, chioft, Moft=1, interp=True):
        """
        Return a list of arrays. Each array in the list contains the time 
        dependant complex frequency (determined by the spin and mass arrays).
        An array is constructed for each mode in the modes list.
        
        Parameters
        ----------            
        modes : array_like
            A sequence of (l,m,n) tuples to specify which QNMs to load 
            frequencies for. For nonlinear modes, the tuple has the form 
            (l1,m1,n1,l2,m2,n2,...).
            
        chioft : array_like
            The dimensionless spin magnitude of the black hole.
            
        Moft : array_like, optional
            The time dependant mass of the black hole. See the qnm.omega() 
            docstring for details on units. This can either be a float, which
            then divides through the whole omega array, or an array of the
            same length as chioft. The default is 1, in which case
            no scaling is performed.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is True.
            
        Returns
        -------
        list
            The list of complex QNM frequencies, where each element in the 
            list is an array of length chioft.
        """
        # Code for linear QNMs:
        # omegas = []
        # for l, m, n, sign in modes:
        #     omegas.append(self.omegaoft(l, m, n, sign, chioft, Moft, interp))
        # return omegas
            
        # Code for nonlinear QNMs:
        return [
            sum([self.omegaoft(l, m, n, sign, chioft, Moft, interp) 
                 for l, m, n, sign in [mode[i:i+4] for i in range(0, len(mode), 4)]
                 ]) 
            for mode in modes
            ]
    