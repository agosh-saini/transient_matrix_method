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
            # Filter the data for the specific sensor
            sensor_data = self.summary_data[self.summary_data['Sensor'] == sensor]['Variation']

            # Remove outliers using IQR
            Q1 = sensor_data.quantile(0.25)  # First quartile
            Q3 = sensor_data.quantile(0.75)  # Third quartile
            IQR = Q3 - Q1  # Interquartile range

            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out outliers
            filtered_data = sensor_data[(sensor_data >= lower_bound) & (sensor_data <= upper_bound)]

            # Return the mean of the filtered data
            return filtered_data.min()

    
    def calculate_lod(self, analyte, concentrations, sensor, tested_std=None, threshold=500):
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

        missing_data_indices = 0

        for concentration in concentrations:
            sorted_data = self._sort_data(analyte, concentration)
            
            if sorted_data.empty:
                missing_data_indices += 1
                warn(f"No data found for analyte '{analyte}' at concentration '{concentration}'.", RuntimeWarning)
                continue

            sensor_data = sorted_data[sorted_data['Sensor'] == sensor]

            if sensor_data.empty:
                warn(f"No data found for sensor '{sensor}' at concentration '{concentration}'.", RuntimeWarning)
                missing_data_indices += 1
                continue
    
            if tested_std is not None:
                std_values.append(tested_std[sensor])
            else:
                std_values.append(sensor_data['Variation'].mean())
            y_values.append(sensor_data['Delta_R'].mean())

        if len(concentrations) < 2:
            warn("At least two concentrations are required for LOD calculation.", RuntimeWarning)
            return np.nan, np.nan
        
        if missing_data_indices > 0:
            warn(f"Missing data for either concentration or sensor.", RuntimeWarning)
            return np.nan, np.nan

        # Fit a line to the data (concentration vs Delta_R)
        slope, _ = np.polyfit(concentrations, y_values, 1)

        if slope == 0:
            raise ZeroDivisionError("Slope is zero; LOD calculation is not possible.")

        # Calculate LOD
        average_std = np.mean(std_values)
        lod = (3 * average_std) / slope

        if lod < 0:
            warn("WARNING: LOD is less than 0. This is not possible.", RuntimeWarning)
            lod, slope = np.nan, np.nan
        
        if lod > threshold:
            warn("WARNING: LOD is greater than the threshold. This is not possible.", RuntimeWarning)
            lod, slope = np.nan, np.nan
        
        return lod, slope
    
    def return_all_lod(self, analytes, concentrations, print_results=False, threshold=500):
        '''
        Return the LOD for all sensors and analytes

        Input:
            analytes: list of analytes to calculate LOD for
            concentrations: list of concentrations to calculate LOD for
            print_results: boolean to print the results

        Output:
            lod_results: dictionary of LOD results
        ''' 
        tested_std = {sensor: self._get_std(sensor) for sensor in self.sensor_list}

        lod_results = {sensor: {} for sensor in self.sensor_list}

        for analyte in analytes:
            for sensor in self.sensor_list:
                lod, slope = self.calculate_lod(analyte, concentrations, sensor, tested_std=tested_std, threshold=threshold)
                lod_results[f"{sensor}"][f"{analyte}"] = {'LOD': lod, 'Std': tested_std[sensor], 'Slope': slope}

        if print_results:
            for sensor in self.sensor_list:
                print(f"{sensor}:")
                for analyte in analytes:
                    print(f"{analyte}: LOD = {lod_results[sensor][analyte]['LOD']:.0f}, Std = {lod_results[sensor][analyte]['Std']:.4E}, Slope = {lod_results[sensor][analyte]['Slope']:.4E}")

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

    lod_results = lod_calculator.return_all_lod(analytes, concentrations, print_results=True, threshold=50)

    # Save the results to a CSV file
    lod_results_df = pd.DataFrame(lod_results)
    lod_results_df.to_csv("LOD_results/LOD_results.csv", index=True)

    keys = []
    lod_values = []
    analyte_values = []

    # plot the results
    for sensor in lod_results:
        for analyte in lod_results[sensor]:
            keys.append(f"{sensor}")
            lod_values.append(lod_results[sensor][analyte]['LOD'])
            analyte_values.append(analyte)

    for i, lod in enumerate(lod_values):
        if lod is np.nan:
            lod_values.remove(lod)
            keys.remove(keys[i])
            analyte_values.remove(analyte_values[i])


    plt.figure(figsize=(10, 6))

    for analyte in set(analyte_values):
        mask = [a == analyte for a in analyte_values]
        analyte_keys = [keys[i] for i in range(len(keys)) if mask[i]]
        analyte_lods = [lod_values[i] for i in range(len(lod_values)) if mask[i]]
        
        plt.scatter(analyte_keys, analyte_lods, label=analyte)

    plt.legend()
    plt.grid(True)

    plt.xlabel('Sensor')
    plt.xticks(rotation=45, fontsize=5)

    plt.ylabel('LOD')

    # Create a more descriptive and formatted title
    concentrations_str = ', '.join(map(str, concentrations))
    title = f'Limit of Detection (LOD) Analysis\n{", ".join(analytes)} at {concentrations_str} ppm'
    plt.title(title, pad=20, fontsize=12, fontweight='bold')

    plt.savefig(f"LOD_results/LOD_results.png")
    plt.show()

