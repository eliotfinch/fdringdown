import numpy as np

# Manipulating file paths
import os

# 1D interpolation
from scipy.interpolate import interp1d

# Vector rotation
from scipy.spatial.transform import Rotation as R

import lal

# Constants
from .utils import c


class interferometer:
    """
    A class to hold functions relating to the interferometers. In particular,
    the amplitude spectral densities (ASDs) associated with each 
    interferometer can be loaded and interpolated, complex waveforms can be 
    projected onto the interferometers, and functions related to time shifting
    and arrival times are also available.
    
    Parameters
    ----------
    name : str
        The name of the desired interferometer. For example, 'H1' for the
        Hanford interferometer.
    """
    
    def __init__(self, name):
        """
        Initialize the class.
        """
        self.name = name
        
        # Dictionary to store the interpolated ASDs
        self.asd_funcs = {}
        
        # The directory of this file (current working directory)
        cwd = os.path.abspath(os.path.dirname(__file__))
        
        # The directory we store the interferometer data (data directory)
        self.dd = f'{cwd}/data/interferometers/'
        
        # Load interferometer arm orientation and location data, see Table 1 
        # of https://arxiv.org/abs/gr-qc/0008066
        self.vertex, nx, ny = np.loadtxt(self.dd+name+'.txt')
        
        # Calculate the detector tensor from the arm orientations (equation B6
        # in the reference)
        self.detector_tensor = 0.5*(
            np.einsum('i,j->ij', nx, nx) - np.einsum('i,j->ij', ny, ny))
        
        
    def load_asd(self, asd_name):
        """
        Load a requested ASD data file from the noise_curved directory.

        Parameters
        ----------
        asd_name : str
            The name of the ASD to load, e.g. 'O3'.
        """
        # Load in the file
        asd_path = f'{self.dd}/noise_curves/{self.name}_{asd_name}.txt'
        
        asd_freqs, asd = np.loadtxt(asd_path).T
        
        # Interpolate 
        asd_interp = interp1d(asd_freqs, asd, bounds_error=False, fill_value=0)
        
        # Store to dictionary
        self.asd_funcs[asd_name] = asd_interp
        
        
    def asd(self, asd_name):
        """
        Get an amplitude spectral density (ASD) function associated with the
        interferometer.

        Parameters
        ----------
        asd_name : str
            The name of the ASD to load, e.g. 'O3'.

        Returns
        -------
        func
            An (interpolated) ASD function that can be evaluated at any 
            frequencies.
        """
        # Load the ASD if it hasn't been already
        if asd_name not in self.asd_funcs:
            self.load_asd(asd_name)
            
        return self.asd_funcs[asd_name]
    
    
    def response(self, signal, ra, dec, t_event, psi, dt):
        """
        Project a signal in the Earth center frame onto a detector, shifting
        in time appropriately.

        Parameters
        ----------
        signal : array
            The complex GW signal.
            
        ra : float
            The right ascension of the signal in radians.
            
        dec : float
            The declination of the signal in radians.
            
        t_event : float
            The GPS time of arrival of the signal.
            
        psi : float
            The polarization angle (counter-clockwise about the direction of
            propagation) in radians.
            
        dt : float
            The time resolution of the signal.

        Returns
        -------
        response : array
            The (real) signal observed in the interferometer, shifted in time.
        """
        # First, project the signal onto the interferometer
        projected_signal = self.project(signal, ra, dec, t_event, psi)
        
        # With the assumption the signal is in an Earth center frame, we 
        # calculate how long it takes the signal to reach the interferometer 
        # from the center of the Earth
        time_delay = self.time_delay([0,0,0], ra, dec, t_event)
        
        # We will need to shift the data by this number of steps
        shift = int(time_delay/dt)
        
        # Perform the shift. np.roll shifts the data to the right by the given 
        # number of steps. If the time delay is positive, then the signal 
        # reaches the detector at a later time than the Earth center. So, 
        # np.roll shifts the data in the correct direction.
        response = np.roll(projected_signal, shift)
        
        return response
        
        
    def project(self, signal, ra, dec, t_event, psi, FD=False):
        """
        Project a signal onto a detector. See appendix B of 
        https://arxiv.org/abs/gr-qc/0008066

        Parameters
        ----------
        signal : array
            The complex GW signal.
            
        ra : float
            The right ascension of the signal in radians.
            
        dec : float
            The declination of the signal in radians.
            
        t_event : float
            The GPS time of arrival of the signal.
            
        psi : float
            The polarization angle (counter-clockwise about the direction of
            propagation) in radians.

        Returns
        -------
        projected_signal : array
            The (real) signal observed in the interferometer.
        """
        # Separate signal into plus and cross polarizations
        if FD:
            signal_polarizations = signal
            
        else:
            signal_polarizations = {
                'plus': np.real(signal),
                'cross': -np.imag(signal)}
            
        # Create an empty array to add the result to
        projected_signal = np.zeros_like(signal_polarizations['plus'])
            
        for polarization in ['plus', 'cross']:
            
            # Calculate the polarization tensor for the given parameters
            polarization_tensor = self.get_polarization_tensor(
                ra, dec, t_event, psi, polarization)
            
            # Calculate the interferometer response (equation B7 in the 
            # reference)
            response = signal_polarizations[polarization]*np.einsum(
                'ij,ij', self.detector_tensor, polarization_tensor)
            
            # Add to the projected signal
            projected_signal += response
    
        return projected_signal
    
    
    def get_polarization_tensor(self, ra, dec, t_event, psi, polarization):
        """
        Calculate the polarization tensor for a given sky location, time of
        arrival and polarization angle. 
        
        For the definition of the Earth fixed coordinate system used, see
        appendix B of https://arxiv.org/abs/gr-qc/0008066. 

        Parameters
        ----------
        ra : float
            The right ascension of the signal in radians.
            
        dec : float
            The declination of the signal in radians.
            
        t_event : float
            The GPS time of arrival of the signal.
            
        psi : float
            The polarization angle (counter-clockwise about the direction of
            propagation) in radians.
            
        polarization : str
            The polarization to consider. Options are 'plus' and 'cross'.

        Returns
        -------
        polarization_tensor : array
            A 3x3 array representation of the polarization tensor.
        """
        # Convert the sky position to spherical polar coordinates in our Earth
        # fixed frame
        gmst = lal.GreenwichMeanSiderealTime(t_event) % (2*np.pi)
        theta = np.pi/2 - dec
        phi = ra - gmst
        
        # Calculate the vectors corresponding to the axes of the wave frame
        X = np.array(
            [  np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta),
             -(np.cos(phi)*np.cos(psi) + np.sin(psi)*np.sin(phi)*np.cos(theta)),
               np.sin(psi)*np.sin(theta)  ])
        
        Y = np.array(
            [-(np.sin(phi)*np.sin(psi) + np.cos(psi)*np.cos(phi)*np.cos(theta)),
               np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.cos(theta),
               np.sin(theta)*np.cos(psi)  ])
        
        # Calculate the polarization tensor
        if polarization == 'plus':
            polarization_tensor = np.einsum('i...,j...->ij...', X, X) \
                - np.einsum('i...,j...->ij...', Y, Y)
            
        elif polarization == 'cross':
            polarization_tensor = np.einsum('i...,j...->ij...', X, Y) \
                + np.einsum('i...,j...->ij...', Y, X)
        
        return polarization_tensor
    
    
    def detector_dt_from_sky_loc(self, IFO_list, ra, dec, t_event):
        """
        Generate a detector_dt dictionary for use with the .get_data_segment()
        method.
        
        The reference interferometer (for which dt=0) is defined as the
        interferometer for which this function is called from.

        Parameters
        ----------
        IFO_list : list
            A list of interferometer objects.
            
        ra : float
            The right ascension of the signal in radians.
            
        dec : float
            The declination of the signal in radians.
            
        t_event : float
            The GPS time of arrival of the signal.

        Returns
        -------
        detector_dt : dict
            A dictionary of signal time delays for each interferometer, with
            respect to the reference interferometer.
            
            The dt value for each detector is defined as 
            (reference detector arrival time) - (detector arrival time).
        """
        # Create the detector_dt dictionary
        detector_dt = {}
        
        for IFO in IFO_list:
            
            # The interferometer for which this function is called is defined
            # to be the reference interferometer - so dt is zero
            if IFO.name == self.name:
                detector_dt[IFO.name] = 0
            
            # Calculate the time delay for each interferometer in the list
            else:
                dt = self.time_delay(IFO.vertex, ra, dec, t_event)
                detector_dt[IFO.name] = dt
                
        return detector_dt
        

    def time_delay(self, vertex, ra, dec, t_event):
        """
        Calculate time delay between this interferometer and a second
        location, for a given signal sky position and arrival time. 
        
        Parameters
        ----------
        vertex : array
            A cartesian coordinate vector in the Earth fixed coordinate system, 
            as described in appendix B of https://arxiv.org/abs/gr-qc/0008066, 
            for which a time delay is calculated. 
            
            For the time delay between this interferometer and a second 
            interferometer, pass the second interferometer's .vertex method.
            For a time delay from the Earth center, pass np.array([0,0,0]). 
            
        ra : float
            The right ascension of the signal in radians.
            
        dec : float
            The declination of the signal in radians.
            
        t_event : float
            The GPS time of arrival of the signal.
    
        Returns
        -------
        time_delay : float
            The time delay in seconds between this detector and the given 
            coordinates. This is defined as
            (detector arrival time) - (vertex arrival time).
    
        """
        # Convert the sky position to spherical polar coordinates in our Earth
        # fixed frame
        gmst = lal.GreenwichMeanSiderealTime(t_event) % (2*np.pi)
        theta = np.pi / 2 - dec
        phi = ra - gmst
        
        # Unit vector pointing in the direction of the signal
        omega = np.array(
            [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        
        # The difference in position
        delta_d = vertex - self.vertex
        
        # Calculate the distance along the direction of travel between the two
        # positions, and convert to a time
        return np.dot(omega, delta_d)/c
    
    
    def sky_location(self, vertex, time_delay, t_event):
        """
        Calculate the possible sky locations of a source, for a given time 
        delay (between this interferometer and a given vertex) and event time.

        Parameters
        ----------
        vertex : array
            A cartesian coordinate vector in the Earth fixed coordinate system, 
            as described in appendix B of https://arxiv.org/abs/gr-qc/0008066, 
            for which a time delay is calculated. 
            
            For the time delay between this interferometer and a second 
            interferometer, pass the second interferometer's .vertex method.
            For a time delay from the Earth center, pass np.array([0,0,0]). 
            
        time_delay : float
            The time delay in seconds between this detector and the given 
            vertex. This is defined as
            (detector arrival time) - (vertex arrival time).
            
        t_event : float
            The GPS time of arrival of the signal.

        Returns
        -------
        ra_array : array
            The possible right ascension values.
            
        dec_array : array
            The possible declination values.
        """
        # The distance between this interferometer and the vertex
        D = np.linalg.norm(vertex - self.vertex)
        
        # Construct the unit vector pointing from one vertex to the other
        if time_delay > 0:
            # If time_delay is positive, then the signal arrives at the 
            # provided vertex first, so we need a unit vector pointing from 
            # vertex to self.vertex
            n = (self.vertex-vertex)/D
        else:
            # If time_delay is negative, flip its sign and use a unit vector
            # pointing from self.vertex to vertex
            time_delay *= -1
            n = (vertex-self.vertex)/D
            
        # The opening angle of the cone
        theta = np.arcsin(c*time_delay/D)
        
        # Get an (arbitrary) vector which is perpendicular to n
        m = np.cross(n, [0,0,1])
        
        # Normalize
        m /= np.linalg.norm(m)
        
        # A vector perpendicular to n and m
        o = np.cross(n,m)
        
        # We rotate m about o by theta (counter-clockwise). This gives us a 
        # possible direction to the source. 
        r = R.from_rotvec(theta*o)
        k = r.apply(m)
        
        # All possible source directions can be obtained by rotating k about n
        angle_array = np.arange(0, 2*np.pi, 0.01)
        
        # We can store all the rotations at once like this
        r = R.from_rotvec(np.outer(angle_array,n))
        
        # Apply
        possible_directions = r.apply(k)
        
        # We must now convert these directions to a right ascension and 
        # declination. First, convert to spherical polars.
        theta_array = np.arccos(possible_directions[:,2])
        phi_array = np.arctan2(
            possible_directions[:,1], possible_directions[:,0])
        
        # Then, convert to right ascension and declination
        gmst = lal.GreenwichMeanSiderealTime(t_event) % (2*np.pi)
        ra_array = phi_array + gmst
        dec_array = np.pi/2 - theta_array
        
        return ra_array, dec_array