######################################################
# Author: Agosh Saini
# Contact: contact@agoshsaini.com
######################################################
# Description: This script is used to find the trend of sensors and sensor groups. It calculates the slope of the linear fit for each sensor and sensor group. The script also plots the trend of sensors and sensor groups.
######################################################

############# Import ############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from scipy import stats

############# Init Value ############

seed = 20785161
np.random.seed(seed)

class_values = dict(zip([900, 1900, 2900], [1,2,3]))  # Map concentrations to classes
values = [1,2,3] # Possible class labels

############# Define Functions ############
def linear(x, m, c):
    '''
        Linear function

        Parameters:
            x (float): x value
            m (float): Slope
            c (float): Intercept

        Returns:
            float: y value
    '''
    return m*x + c

def remove_outliers(data):
    '''
        Remove outliers from the data

        Parameters:
            data (array): Data to be processed

        Returns:
            array: Processed data
    '''
    z_scores = stats.zscore(data)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=0)

    return data[filtered_entries]

############# Load Data ############

data = pd.read_csv("summary_output/summary.csv")  

# Map concentration values to class labels
data["Conc"] = data["Conc"].map(class_values)

############# Prepare Features ############

# Create a unique identifier for each sample - File name is unique for each sample and can also be used
data['Sample ID'] = data.index 

# Pivot the dataset to aggregate sensor data
pivoted_data = data.pivot(index="Sample ID", columns="Sensor ID")[["Delta R On", "Delta R Off"]].reset_index()

# Add the class labels for each sample
pivoted_data["Conc"] = data.groupby("Sample ID")["Conc"].first().values

# Fill missing values with 0 (or a suitable constant)
pivoted_data = pivoted_data.fillna(0)

############# Create Df for Saving Data ############

df = pd.DataFrame(columns=['Conc', 'Sensor ID', 'Delta R On', 'Delta R On Std', 'Delta R Off', 'Delta R Off Std'])

df_sensors = pd.DataFrame(columns=['Sensor ID', 'Delta R On', 'Delta R On Std', 'Delta R Off', 'Delta R Off Std'])

############# Sorting and Processing Data ############

# Repeat for every concentration

for conc in data['Conc'].unique():

    # Filter data for the current concentration
    conc_pivoted_data = pivoted_data[pivoted_data['Conc'] == conc]

    # Create a dictionary to store the standard deviation and mean for each sensor
    sub_sensor_variability = {i: [] for i in data['Sensor ID'].unique()}

    # Repeat for both Delta R On and Delta R Off
    for cycle in ['Delta R On', 'Delta R Off']:

        # Create a dictionary to store the standard deviation and mean for each sensor group
        sensor_variability = {i.split('.')[0]: [] for i in data['Sensor ID'].unique()} 
            
        # Repeat for each sensor
        for sensor in sub_sensor_variability.keys(): 

            std_dev = conc_pivoted_data[(cycle, sensor)].std()
            mean_val = conc_pivoted_data[(cycle, sensor)].mean()

            # Append standard deviation and mean to the list
            sub_sensor_variability[sensor].append([std_dev, mean_val, conc])

        # Repeat for each sensor group for finding general trend of sensors
        for sensor in sub_sensor_variability.keys():

            sensor_group = sensor.split('.')[0] 
            
            # Append standard deviation and mean to the list
            std_dev = conc_pivoted_data[(cycle, sensor)].std()
            mean_val = conc_pivoted_data[(cycle, sensor)].mean()

            sensor_variability[sensor_group].append([std_dev, mean_val])

        # Calculate the average standard deviation and mean for each sensor group
        for sensor_group in sensor_variability.keys():

            std_dev = [i[0] for i in sensor_variability[sensor_group]]
            mean_val = [i[1] for i in sensor_variability[sensor_group]]
            
            sensor_variability[sensor_group] = [np.mean(std_dev), np.mean(mean_val)]

        # Append the data to the dataframe
        if cycle == 'Delta R On':

            rows = []

            for sensor in sensor_variability.keys():

                rows.append({'Conc': conc,
                            'Sensor ID': sensor, 
                            'Delta R On': sensor_variability[sensor][1], 
                            'Delta R On Std': sensor_variability[sensor][0], 
                            'Delta R Off': 'N/A',
                            'Delta R Off Std': 'N/A'})

                if rows:
                    df_sensors = pd.concat([df_sensors, pd.DataFrame(rows)], ignore_index=True)
                else:
                    print('No data for concentration: ', conc)

        else:

            for sensor in sensor_variability.keys():

                df_sensors.loc[df_sensors['Sensor ID'] == sensor, 'Delta R Off'] = sensor_variability[sensor][1]
                df_sensors.loc[df_sensors['Sensor ID'] == sensor, 'Delta R Off Std'] = sensor_variability[sensor][0]



    rows = []

    # Append the data to the dataframe
    for sensor in sub_sensor_variability.keys():

        rows.append({'Conc': sub_sensor_variability[sensor][0][2], 
                        'Sensor ID': sensor, 
                        'Delta R On': sub_sensor_variability[sensor][0][1], 
                        'Delta R On Std': sub_sensor_variability[sensor][0][0],
                        'Delta R Off': sub_sensor_variability[sensor][1][1],
                        'Delta R Off Std': sub_sensor_variability[sensor][1][0]})
    
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True).drop_duplicates()
    else:
        print('No data for concentration: ', conc)

############# Plotting ############

# Create a directory to save the plots
if os.path.exists('sensor_trend') == False:
    os.mkdir('sensor_trend')

# Sort the dataframe by Sensor ID
df = df.sort_values(by='Sensor ID')

# Calculate the slope on of the linear fit for each sensor
for sensor in df['Sensor ID'].unique():

    sensor_df = df[df['Sensor ID'] == sensor]

    x = sensor_df['Conc']
    y = sensor_df['Delta R On']

    popt, _ = curve_fit(linear, x, y)

    df.loc[df['Sensor ID'] == sensor, 'Slope On'] = popt[0]

# Plot the trend of all sensors
x, y = [], []

for sensor in df['Sensor ID'].unique():
    x.append(sensor)
    y.append(df[df['Sensor ID'] == sensor]['Slope On'].mean())

plt.figure(figsize=(10, 6))
plt.scatter(x,y)
plt.xlabel('Sensor ID')
plt.xticks(rotation=90)
plt.ylabel('Slope On')
plt.title('All Sensors Trend - On Cycle')
plt.savefig('sensor_trend/all_sensors_on.png')
plt.show()

# Calculate the slope off of the linear fit for each sensor
for sensor in df['Sensor ID'].unique():

    sensor_df = df[df['Sensor ID'] == sensor]

    x = sensor_df['Conc']
    y = sensor_df['Delta R Off']

    popt, _ = curve_fit(linear, x, y)

    df.loc[df['Sensor ID'] == sensor, 'Slope Off'] = popt[0]

# Plot the trend of all sensors
x, y = [], []

for sensor in df['Sensor ID'].unique():
    x.append(sensor)
    y.append(df[df['Sensor ID'] == sensor]['Slope Off'].mean())

plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.xlabel('Sensor ID')
plt.xticks(rotation=90)
plt.ylabel('Slope Off')
plt.title('All Sensors Trend - Off Cycle') 
plt.savefig('sensor_trend/all_sensors_off.png')
plt.show()



# Check group response for sensors during On Cycle
x, y, err = [], [], []

for sensor in df['Sensor ID'].unique():

    if sensor.split('.')[0] not in x:
        x.append(sensor.split('.')[0])

        # Remove outliers
        y_temp = df[df['Sensor ID'].str.contains(sensor.split('.')[0])]['Slope On'].values
        y_temp = remove_outliers(y_temp)

        y.append(np.mean(y_temp))
        err.append(np.std(y_temp))

plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.errorbar(x, y, yerr=err, fmt='o', ecolor='r', capthick=2)
plt.xlabel('Sensor Group Response - On Cycle') 
plt.xticks(rotation=90)
plt.ylabel('Slope On')
plt.title('Sensor Group Trend')
plt.savefig('sensor_trend/sensor_group_on.png')
plt.show()


# Check group response for sensors during Off Cycle
x, y, err = [], [], []

for sensor in df['Sensor ID'].unique():

    if sensor.split('.')[0] not in x:
        x.append(sensor.split('.')[0])

        # Remove outliers
        y_temp = df[df['Sensor ID'].str.contains(sensor.split('.')[0])]['Slope Off'].values
        y_temp = remove_outliers(y_temp)

        y.append(np.mean(y_temp))
        err.append(np.std(y_temp))

plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.errorbar(x, y, yerr=err, fmt='o', ecolor='r', capthick=2)
plt.xlabel('Sensor Group Response - Off Cycle')
plt.xticks(rotation=90)
plt.ylabel('Slope Off')
plt.title('Sensor Group Trend')
plt.savefig('sensor_trend/sensor_group_off.png')
plt.show()

############# Save Data ############
path = 'summary_output'

# Save the data
df.to_csv(f'{path}/sensor_trend.csv', index=False)
df_sensors.to_csv(f'{path}/subsensor_trends.csv', index=False)

