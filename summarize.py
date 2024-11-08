######################################################
# Author: Agosh Saini
# Date: 2024-10-25
# Contact: contact@agoshsaini.com
######################################################
# Description: This script summarizes the data from JSON files
######################################################

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

class DataSummary:

    def __init__(self, data_dir):
        '''
        Initialize the class with the directory containing JSON files.

        Parameters:
            data_dir: directory containing JSON files
        '''
        self.data_dir = data_dir
        self.data = self.load_data()

    def load_data(self):
        '''
        Load data from JSON files in the specified directory.

        Returns:
            data: dictionary containing loaded data
        '''
        data = {}
        for file in os.listdir(self.data_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.data_dir, file)) as f:
                    data[file] = json.load(f)
        return data

    def summarize_data(self):
        '''
        Summarize the data from the loaded JSON files.

        Returns:
            summary: dictionary containing summarized data
        '''
        summary = {}
        for file, content in self.data.items():
            summary[file] = {
                'Filename': content['filename'],
                'Analyte': content['Analyte'],
                'Material': content['Material'],

                'Conc': content['ppm'],

                'Sensor ID': content['Sensor Type'],
                
                'Delta R On': content['RC_on']['Delta_R'],
                'Tau On': content['RC_on']['tau'],

                'Delta R Off': content['RC_off']['Delta_R'],
                'Tau Off': content['RC_off']['tau'],

            }
        return summary
    
    def generate_csv(self, output_dir):
        '''
        Generate a CSV file from the summarized data.

        Parameters:
            output_dir: directory to save the CSV file
        '''
        summary = self.summarize_data()
        df = pd.DataFrame(summary).T
        df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
        print('CSV file generated successfully!')

    def split_csv_ppm(self, output_dir):
        '''
        Split the CSV file based on the concentration.

        Parameters:
            output_dir: directory to save the split CSV files
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        for conc in df['Conc'].unique():
            df_conc = df[df['Conc'] == conc]
            df_conc.to_csv(os.path.join(output_dir, f'summary_{conc}.csv'), index=False)
        print('CSV files split successfully!')

    def plot_delta_r_per_sensor(self, output_dir):
        '''
        Plot the Delta R values for each sensor with all concentrations in one figure.
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        for sensor in df['Sensor ID'].unique():
            plt.figure()
            for conc in df['Conc'].unique():
                df_conc = df[(df['Conc'] == conc) & (df['Sensor ID'] == sensor)]
                plt.scatter([sensor]*len(df_conc), df_conc['Delta R On'], label=f'{conc} ppm')
            plt.xlabel('Sensor ID')
            plt.ylabel('Delta R On')
            plt.title(f'Delta R On vs Concentration for Sensor {sensor}')
            plt.legend(title='Concentration (ppm)')
            plt.savefig(os.path.join(output_dir, f'delta_r_sensor_{sensor}.png'))
            plt.close()

    def plot_delta_r_grouped_by_prefix(self, output_dir):
        '''
        Plot the Delta R values for each sensor prefix with all concentrations in one figure
        and save a large, high-resolution version of the image.
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        df['Sensor Prefix'] = df['Sensor ID'].apply(lambda x: x.split('.')[0])  # Extract the prefix

        for prefix in df['Sensor Prefix'].unique():
            plt.figure(figsize=(16, 10))  # Set a large figure size for high resolution
            
            df_prefix = df[df['Sensor Prefix'] == prefix]
            
            for conc in df_prefix['Conc'].unique():
                df_conc = df_prefix[df_prefix['Conc'] == conc]
                plt.scatter(df_conc['Sensor ID'], df_conc['Delta R On'], label=f'{conc} ppm')

            plt.xlabel('Sensor ID')
            plt.ylabel('Delta R On')
            plt.title(f'Delta R On for Sensors with Prefix {prefix}')
            plt.yscale('log')
            plt.legend(title='Concentration (ppm)')
            plt.savefig(os.path.join(output_dir, f'delta_r_on_{prefix}.png'), dpi=300)  # Save with high DPI for better quality
            plt.close()

    def plot_tau_grouped_by_prefix(self, output_dir):
        '''
        Plot the Delta R values for each sensor prefix with all concentrations in one figure
        and save a large, high-resolution version of the image.
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        df['Sensor Prefix'] = df['Sensor ID'].apply(lambda x: x.split('.')[0])  # Extract the prefix

        for prefix in df['Sensor Prefix'].unique():
            plt.figure(figsize=(16, 10))  # Set a large figure size for high resolution
            
            df_prefix = df[df['Sensor Prefix'] == prefix]
            
            for conc in df_prefix['Conc'].unique():
                df_conc = df_prefix[df_prefix['Conc'] == conc]
                plt.scatter(df_conc['Sensor ID'], df_conc['Tau On'], label=f'{conc} ppm')

            plt.xlabel('Sensor ID')
            plt.ylabel('Delta R On')
            plt.title(f'Delta R On for Sensors with Prefix {prefix}')
            plt.yscale('log')
            plt.legend(title='Concentration (ppm)')
            plt.savefig(os.path.join(output_dir, f'tau_on_{prefix}.png'), dpi=300)  # Save with high DPI for better quality
            plt.close()





if __name__ == '__main__':
    data_dir = 'json_folder'
    output_dir = 'summary_output'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds = DataSummary(data_dir)
    ds.generate_csv(output_dir)
    print('Data summarized successfully')

    ds.split_csv_ppm(output_dir)
    print('CSV files split successfully')

    ds.plot_delta_r_grouped_by_prefix(output_dir)
    print('Delta R plots generated successfully')

    ds.plot_tau_grouped_by_prefix(output_dir)
    print('Tau plots generated successfully')