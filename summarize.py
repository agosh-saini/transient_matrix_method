######################################################
# Author: Agosh Saini
# Date: 2024-10-25
# Contact: contact@agoshsaini.com
######################################################
# Description: This script summarizes the data from JSON files
######################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import json

class DataSummary:

    def __init__(self, data_dir):
        '''
        Initialize the class with the directory containing JSON files.
        '''
        self.data_dir = data_dir
        self.data = self.load_data()

    def load_data(self):
        '''
        Load data from JSON files in the specified directory.
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
        '''
        summary = {}
        for file, content in self.data.items():
            summary[file] = {
                'Filename': content['filename'],
                'Analyte': content['Analyte'][0],   # Extract single element from list
                'Material': content['Material'][0], # Extract single element from list
                'Conc': content['ppm'],
                'Sensor ID': content['Sensor Type'],
                'Delta R On': content['RC_on']['Delta_R'],
                'Tau On': content['RC_on']['tau'],
                'R0 On': content['RC_on']['R0'],
                'Delta R Off': content['RC_off']['Delta_R'],
                'Tau Off': content['RC_off']['tau'],
                'R0 Off': content['RC_off']['R0']
            }
        return summary
    
    def generate_csv(self, output_dir):
        '''
        Generate a CSV file from the summarized data.
        '''
        summary = self.summarize_data()
        df = pd.DataFrame(summary).T
        df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
        print('CSV file generated successfully!')

    def split_csv_ppm(self, output_dir):
        '''
        Split the CSV file based on the concentration.
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        df['Filename_Prefix'] = df['Filename'].apply(lambda x: x.split("_")[0])
        for conc in df['Conc'].unique():
            file_name = f'{df["Filename_Prefix"].iloc[0]}_{df["Analyte"].iloc[0]}_{df["Material"].iloc[0]}_summary_{conc}.csv'
            df_conc = df[df['Conc'] == conc]
            df_conc.to_csv(os.path.join(output_dir, file_name), index=False)
        print('CSV files split successfully!')

    def plot_delta_r_grouped_by_prefix(self, output_dir):
        '''
        Plot the Delta R values for each sensor prefix with all concentrations in one figure
        and save a large, high-resolution version of the image.
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        df['Sensor Prefix'] = df['Sensor ID'].apply(lambda x: x.split('.')[0])
        df['Filename_Prefix'] = df['Filename'].apply(lambda x: x.split("_")[0])

        for prefix in df['Sensor Prefix'].unique():
            plt.figure(figsize=(16, 10))
            file_name = f'{df["Filename_Prefix"].iloc[0]}_{df["Analyte"].iloc[0]}_{df["Material"].iloc[0]}_delta_r_on_{prefix}.png'
            
            df_prefix = df[df['Sensor Prefix'] == prefix]
            
            for conc in df_prefix['Conc'].unique():
                df_conc = df_prefix[df_prefix['Conc'] == conc]
                plt.scatter(df_conc['Sensor ID'], (df_conc['R0 On'] + df_conc['Delta R On'])/df_conc['R0 On'], label=f'{conc} ppm')

            plt.xlabel('Sensor ID')
            plt.ylabel('Delta R On')
            plt.title(f'Delta R On for Sensors with Prefix {prefix}')
            #plt.yscale('log')
            plt.legend(title='Concentration (ppm)')
            plt.savefig(os.path.join(output_dir, file_name), dpi=300)
            plt.close()

    def plot_tau_grouped_by_prefix(self, output_dir):
        '''
        Plot the Tau On values for each sensor prefix with all concentrations in one figure
        and save a large, high-resolution version of the image.
        '''
        df = pd.read_csv(os.path.join(output_dir, 'summary.csv'))
        df['Sensor Prefix'] = df['Sensor ID'].apply(lambda x: x.split('.')[0])
        df['Filename_Prefix'] = df['Filename'].apply(lambda x: x.split("_")[0])

        for prefix in df['Sensor Prefix'].unique():
            plt.figure(figsize=(16, 10))
            file_name = f'{df["Filename_Prefix"].iloc[0]}_{df["Analyte"].iloc[0]}_{df["Material"].iloc[0]}_tau_on_{prefix}.png'
            
            df_prefix = df[df['Sensor Prefix'] == prefix]
            
            for conc in df_prefix['Conc'].unique():
                df_conc = df_prefix[df_prefix['Conc'] == conc]
                plt.scatter(df_conc['Sensor ID'], df_conc['Tau On'], label=f'{conc} ppm')

            plt.xlabel('Sensor ID')
            plt.ylabel('Tau On')  # Corrected label
            plt.title(f'Tau On for Sensors with Prefix {prefix}')
            plt.yscale('log')
            plt.legend(title='Concentration (ppm)')
            plt.savefig(os.path.join(output_dir, file_name), dpi=300)
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