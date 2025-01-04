#######
#  Author: Agosh Saini
# Contact: contact@agoshsaini.com   
#######
# This file is a class for LOD calculations
#######

###### IMPORTS ######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from warnings import warn


###### CLASS DEFINITION ######

class LODCalculations:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.sensor_list = self._get_sensor_list()
        self.summary_data = pd.DataFrame(self._get_all_data())

    def _get_sensor_list(self):
        '''
        Get the list of sensors from the data folder
        '''
        sensor_set = set()

        # Iterate through the JSON files in the data folder
        for file in os.listdir(self.data_folder):
            if file.endswith(".json"):
                with open(os.path.join(self.data_folder, file), 'r') as f:
                    data = json.load(f)
                    sensor_set.add(data.get('Sensor Type', 'Unknown'))

        return list(sensor_set)
    
    def _get_all_data(self):
        '''
        Load all data from JSON files in the folder

        Input:

        Output:
            
        '''
        data_array = []

        for file in os.listdir(self.data_folder):
            if file.endswith(".json"):
                with open(os.path.join(self.data_folder, file), 'r') as f:
                    data = json.load(f)
                    data_dict = {
                        'Analyte': data.get('Analyte', 'Unknown')[0],
                        'Concentration': data.get('ppm', np.nan),
                        'Delta_R': data.get('RC_on', {}).get('Delta_R', np.nan),
                        'Variation': data.get('Variation', np.nan),
                        'Sensor': data.get('Sensor Type', 'Unknown')
                    }
                    data_array.append(data_dict)

        return data_array

    def _sort_data(self, analyte, concentration):
        '''
        Filter data by analyte and concentration

        Input:
            analyte: analyte of interest
            concentration: concentration of interest

        Output:
            Summary data which is sorted
        '''
        return self.summary_data[
            (self.summary_data['Analyte'] == analyte) &
            (self.summary_data['Concentration'] == concentration)
        ]
    
    def _get_std(self, sensor, tested_std=None):
        '''
        Get the standard deviation for a sensor

        Input:
            sensor: sensor to get the standard deviation for
            tested_std: dictionary of tested standard deviations

        Output:
            standard deviation for the sensor
        '''
        if tested_std is not None:
            return tested_std[sensor]
        else:
            return self.summary_data[self.summary_data['Sensor'] == sensor]['Variation'].min()
    
    def calculate_lod(self, analyte, concentrations, sensor, tested_std=None):
        '''
        Calculate the LOD for a specific sensor, analyte, and concentration range

        Input:
            analyte: analyte to calculate LOD for
            concentrations: list of concentrations to calculate LOD for
            sensor: sensor to calculate LOD for

        Output:
            lod: LOD for the given analyte, concentration, and sensor
        '''
        y_values = []
        std_values = []

        for concentration in concentrations:
            sorted_data = self._sort_data(analyte, concentration)
            
            if sorted_data.empty:
                raise ValueError(f"No data found for analyte '{analyte}' at concentration '{concentration}'.")

            sensor_data = sorted_data[sorted_data['Sensor'] == sensor]

            if sensor_data.empty:
                raise ValueError(f"No data found for sensor '{sensor}' at concentration '{concentration}'.")

            if tested_std is not None:
                std_values.append(tested_std[sensor])
            else:
                std_values.append(sensor_data['Variation'].mean())
            y_values.append(sensor_data['Delta_R'].mean())

        if len(concentrations) < 2:
            raise ValueError("At least two concentrations are required for LOD calculation.")

        # Fit a line to the data (concentration vs Delta_R)
        slope, _ = np.polyfit(concentrations, y_values, 1)

        if slope == 0:
            raise ZeroDivisionError("Slope is zero; LOD calculation is not possible.")

        # Calculate LOD
        average_std = np.mean(std_values)
        lod = (3 * average_std) / slope

        if lod < 0:
            warn("WARNING: LOD is less than 0. This is not possible.", RuntimeWarning)
        
        return lod, slope
    
    def return_all_lod(self, analytes, concentrations, print_results=False):
        '''
        Return the LOD for all sensors and analytes

        Input:
            analytes: list of analytes to calculate LOD for
            concentrations: list of concentrations to calculate LOD for
            print_results: boolean to print the results

        Output:
            lod_results: dictionary of LOD results
        ''' 
        lod_results = {}

        tested_std = {sensor: self._get_std(sensor) for sensor in self.sensor_list}

        for analyte in analytes:
            for sensor in self.sensor_list:
                lod, slope = self.calculate_lod(analyte, concentrations, sensor, tested_std=tested_std)
                lod_results[f"{analyte}_{sensor}"] = {'LOD': lod, 'Std': tested_std[sensor], 'Slope': slope}

        if print_results:
            for key, value in lod_results.items():
                print(f"{key}: LOD = {value['LOD']:.0f}, Std = {value['Std']:.4E}, Slope = {value['Slope']:.4E}")

        return lod_results


###### MAIN ######

if __name__ == "__main__":
    # Make folder for LOD results
    if not os.path.exists("LOD_results"):
        os.makedirs("LOD_results")
    
    # Define data folder, analyte, and concentration
    data_folder = "json_folder"
    analytes = ['IPA', 'Ace', 'EtOH']
    concentrations = [250, 350, 450]   

    lod_calculator = LODCalculations(data_folder)

    lod_results = lod_calculator.return_all_lod(analytes, concentrations, print_results=True)

    # Save the results to a CSV file
    lod_results_df = pd.DataFrame(lod_results)
    lod_results_df.to_csv("LOD_results/LOD_results.csv", index=True)

    # plot the results
    keys = lod_results.keys()
    lod_values = [lod_results[key]['LOD'] for key in keys]

    plt.figure(figsize=(10, 6))
    plt.scatter(keys, lod_values, label='LOD')
    plt.grid(True)

    plt.xlabel('Analyte_Sensor')
    plt.xticks(rotation=45, fontsize=5)

    plt.ylabel('LOD')

    plt.title(f'LOD Results - {analytes} at {concentrations} ppm')

    plt.savefig(f"LOD_results/LOD_results.png")
    plt.show()

