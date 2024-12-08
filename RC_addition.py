######################################################
# Author: Agosh Saini
# Date: 2024-10-25
# Contact: contact@agoshsaini.com
######################################################
# Description: This script extracts data from json files and then adds RC model to the data
######################################################

######### IMPORTS #########
import numpy as np
import json
import os

from RC_fit import RC_fit

######### CLASS DEFINITION #########
class RC_addition:

    def __init__(self, filepath, graph_path='graph_folder'):
        '''
        Initialize the class with the filepath of the json file.
        '''
        self.filepath = filepath
        self.graph_path = graph_path
    

    def get_values(self, file_name=None):
        '''
        Get the values from the json file.
        '''
        if file_name is None:
            raise ValueError("File name not provided")

        try: 
            with open(file_name, 'r') as file:
                data = json.load(file)

            self.on = np.array(data['ON'])
            self.off = np.array(data['OFF'])
            ''' Remove the baseline if not needed '''
            '''
            self.baseline = np.array(data['Baseline'])
            '''
            self.timestep = int(data['timestep'])
            self.file_name = str(data['filename'])

            return self.on, self.off, self.timestep, self.file_name
        
        except FileNotFoundError:
            raise ValueError(f"File {file_name} not found")

    def get_RC_values(self):
        '''
        Fit the RC model and get the RC values.
        '''
        try: 
            rc = RC_fit(on=self.on, off=self.off, timestep=self.timestep)
            rc.fit()
            self.RC_value = rc.get_parameters()
            return self.RC_value
        
        except Exception as e:
            raise ValueError("Error in fitting the RC model") from e

    def add_RC_values(self, RC_values):
        '''
        Add the RC values to the json file.
        '''
        try: 
            with open(self.filepath, 'r') as file:
                data = json.load(file)

            data['RC_on'] = RC_values['on']
            data['RC_off'] = RC_values['off']

            with open(self.filepath, 'w') as file:
                json.dump(data, file, indent=4)

        except Exception as e:
            raise ValueError(f"Failed to add RC values to {self.filepath}") from e

    def plot_graph(self, visualize={'show': False, 'save': True}):
        '''
        Plot the graph of the RC model.
        '''
        try: 
            rc = RC_fit(on=self.on, off=self.off, timestep=self.timestep)
            rc.fit()
            rc.plot(save_plot=True, file_name=self.file_name, graph_folder=self.graph_path)
        
        except Exception as e:
            raise ValueError("Error in plotting the graph") from e
           
    

    def run(self):
        '''
        Run the class to get values, fit the RC model, plot, and add RC values to json.
        '''
        self.get_values(file_name=self.filepath)
        RC_values = self.get_RC_values()
        self.plot_graph(visualize={'show': False, 'save': True})
        self.add_RC_values(RC_values)



#### EXAMPLE USAGE ####
if __name__ == '__main__':
    folder_path = 'json_folder'
    graph_path = 'graph_folder'

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} not found")

    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    
    for i, file in enumerate(os.listdir(folder_path)):
        rc_addition = RC_addition(f'{folder_path}/{file}', graph_path=graph_path)
        rc_addition.run()
        print(f'RC values added to {file} - File {i+1}/{len(os.listdir(folder_path))}')
