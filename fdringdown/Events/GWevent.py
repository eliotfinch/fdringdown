import numpy as np

# The base class containing data processing functions
from .Event import EventClass

# Locating event data
from gwosc import datasets
from gwosc import locate
from gwosc.api import fetch_event_json

# Downloading files
import urllib.request

# Manipulating file paths
import os

# Saving/loading data
import json

# Reading GW data files
import h5py

# Interferometer class
from ..interferometer import interferometer


class GWevent(EventClass):
    
    def __init__(self, name, sample_rate=4096):
        """
        Initialize the class.
        """
        self.name = name
        self.fs = sample_rate
            
        # Download the data, if it hasn't been already
        self.download_data()
        
        # Load and store data
        self.load_data()
            
    
    def download_data(self):
        
        # The directory of this file (current working directory)
        self.cwd = os.path.abspath(os.path.dirname(__file__))
        
        # The directory we store the GW event data (data directory)
        self.dd = os.path.abspath(
            self.cwd + '/../data/GW_events/' + self.name)
        
        # If there's no internet (or the gwosc api isn't responding) we would
        # like to load previously downloaded data. Use a try block to catch
        # any network errors.
        try:
            
            # Create the data directory if it doesn't exist
            if not os.path.isdir(self.cwd + '/../data/GW_events/'):
                os.mkdir(self.cwd + '/../data/GW_events/')
            if not os.path.isdir(self.dd):
                os.mkdir(self.dd)
                
            # Get the detectors that observed this event
            self.IFO_names = datasets.event_detectors(self.name)

            # Lets ignore G1 for now
            if 'G1' in self.IFO_names:
                self.IFO_names.remove('G1')
            
            # Save the set of IFO names
            with open(self.dd + '/IFO_names.json', 'w') as f:
                json.dump(list(self.IFO_names), f)
                
            # URLs for the data. We default to the larger 4096s datasets.
            urls = locate.get_event_urls(
                self.name, duration=4096, sample_rate=self.fs)
            
            for url in urls:
                
                # The file name
                fn = url.split('/')[-1]
                
                # Download the file if it hasn't been already
                if not os.path.isfile(self.dd + '/' + fn):
                    print('Downloading file', fn)
                    urllib.request.urlretrieve(
                        url, filename=self.dd + '/' + fn)
                    
            # Download the event json
            event_json = fetch_event_json(self.name)['events']
            
            # Access the highest level key (which is just the event name with 
            # a version), and store
            self.parameters = event_json[list(event_json.keys())[0]]
            
            # Save the parameters
            with open(self.dd + '/parameters.json', 'w') as f:
                json.dump(self.parameters, f, indent=4)
            
        except:
            
            print('Could not download data. Attempting to load from file.')
            
            # Get the detectors that observed this event
            with open(self.dd + '/IFO_names.json', 'r') as f:
                self.IFO_names = json.load(f)
                
            # Get the event parameters
            with open(self.dd + '/parameters.json', 'r') as f:
                self.parameters = json.load(f)
            
        # Create a list of interferometer objects
        self.IFO_list = []
        
        for IFO_name in self.IFO_names:
            self.IFO_list.append(interferometer(IFO_name))
                
    
    def load_data(self):
        
        # Dictionary to store the data for each interferometer
        self.data = {}
        
        for IFO_name in self.IFO_names:
            
            # Load the hdf5 file for this interferometer. First, get a list of
            # all files in the data directory.
            fns = os.listdir(self.dd)
            
            # Load the data if the file name conatins the correct IFO name and
            # sampling rate
            for fn in fns:
                if IFO_name in fn:
                    if f'{int(self.fs/1000)}KHZ' in fn:
                        print(f'Loading file {fn}')
                        file = h5py.File(self.dd + '/' + fn, 'r')
                    
            # The strain data
            strain = file['strain']['Strain']
            
            # Store to dictionary
            self.data[IFO_name] = np.array(strain)
        
        # Start time, time resolution and number of samples are shared between 
        # interferometers
        self.start_time = strain.attrs['Xstart']
        self.dt = strain.attrs['Xspacing']
        self.N = strain.attrs['Npoints']
        
        # Construct time array
        self.time = np.linspace(
            self.start_time, self.start_time+4096, num=self.N, endpoint=False)
            

            
                    
            
                    