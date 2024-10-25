######################################################
# Author: Agosh Saini
# Date: 2024-10-25
# Contact: contact@agoshsaini.com
######################################################
# Description: This script extracts data from json files and then adds RC model to the data
######################################################

######### IMPORTS #########
import pandas as pd
import numpy as np
import json
import os

from RC_fit import RC_fit

######### CLASS DEFINITION #########
class RC_addition:

    def __init__(self,filepath):

        '''
        Initialize the class with the filepath of the json file.
        '''

        self.filepath = filepath
    

    def get_values(self, file_name=None):

        '''
        Get the values from the json file.

        Parameters:
            file_name: filepath of the json file (optional)

        Returns:
            on: numpy array for 'on' cycle resistance data
            off: numpy array for 'off' cycle resistance data
            baseline: numpy array for baseline resistance data
            timestep: average timestep between data points
        '''

        if file_name is None:
            file_name = self.filepath

        try: 
            with open(file_name, 'r') as file:
                data = json.load(file)

            self.on = np.array(data['ON'])
            self.off = np.array(data['OFF'])
            self.baseline = np.array(data['Baseline'])
            self.timestep = int(data['timestep'])

            return self.on, self.off, self.baseline, self.timestep
        
        except:
            ValueError("File not found")

    def get_RC_values(self, file_name=None):

        '''
        Get the RC values from the data.

        Parameters:
            file_name: filepath of the json file (optional)

        Returns:
            RC_values: dictionary containing the RC values
        '''

        if file_name is None:
            file_name = self.filepath
        
        try: 
            rc = RC_fit(on=self.on, off=self.off, baseline=self.baseline, timestep=self.timestep)

            rc.fit()

            self.RC_value = rc.get_parameters()

            return rc.get_parameters()
        
        except:
            ValueError("Error in fitting the RC model")

        

    def add_RC_values(self, RC_values, file_name=None):

        '''
        Add the RC values to the json file.

        Parameters:
            RC_values: dictionary containing the RC values
            file_name: filepath of the json file (optional
        '''

        if file_name is None:
            file_name = self.filepath

        try: 
            with open(file_name, 'r') as file:
                data = json.load(file)

            data['RC_on'] = RC_values['on']
            data['RC_off'] = RC_values['off']

            with open(file_name, 'w') as file:
                json.dump(data, file, indent=4)

        except:
            ValueError("Failed to open {file_name}")

    def run(self):

        '''
        Run the class to get the values, fit the RC model, and add the RC values to the json file.
        '''

        self.get_values()

        self.get_RC_values()

        self.add_RC_values(self.RC_value)




#### EXAMPLE USAGE ####
if __name__ == '__main__':
    folder_path = 'json_folder'

    for file in os.listdir(folder_path):
        rc_addition = RC_addition(f'{folder_path}/{file}')
        rc_addition.run()
        

