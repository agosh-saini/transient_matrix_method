import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Create output directory
output_dir = 'PCA_results'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv("summary_output/summary.csv")
print(f"Loaded {len(data)} measurements")

# Function to remove extreme outliers (3x IQR instead of 2x for less strict filtering)
def remove_extreme_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 10 * IQR  # Changed from 2x to 3x IQR
    upper_bound = Q3 + 10 * IQR  # Changed from 2x to 3x IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    if len(outliers) > 0:
        print(f"\nExtreme outliers found for {df['Analyte'].iloc[0]} at {df['Conc'].iloc[0]}ppm:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Outlier values: {outliers[column].values}")
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers and organize data by PN sensors
print("\nProcessing data...")
clean_data = pd.DataFrame()
pn_sensors = sorted([sensor for sensor in data['Sensor ID'].unique() if 'PN' in str(sensor)])
print(f"Found PN sensors: {pn_sensors}")

# Remove outliers for each analyte-concentration combination
for analyte in data['Analyte'].unique():
    for conc in data['Conc'].unique():
        subset = data[(data['Analyte'] == analyte) & (data['Conc'] == conc)].copy()
        if not subset.empty:
            clean_subset = remove_extreme_outliers(subset, 'Delta R On')
            clean_data = pd.concat([clean_data, clean_subset])

print(f"\nOriginal data points: {len(data)}")
print(f"Data points after outlier removal: {len(clean_data)}")

# Prepare combined sensor data for PCA
X_array = []
analyte_labels = []
conc_labels = []

# Group by Analyte and Concentration
for (analyte, conc), group in clean_data.groupby(['Analyte', 'Conc']):
    # Get all measurements for this combination
    sensor_data = {}
    for sensor in pn_sensors:
        sensor_values = group[group['Sensor ID'] == sensor]['Delta R On'].values
        if len(sensor_values) > 0:
            sensor_data[sensor] = sensor_values
    
    # Print debug information
    print(f"\nAnalyte: {analyte}, Concentration: {conc}")
    print(f"Number of measurements per sensor:")
    for sensor, values in sensor_data.items():
        print(f"{sensor}: {len(values)}")
    
    # Proceed if we have at least 3 sensors with data
    if len(sensor_data) >= 3:
        # Get minimum number of measurements across available sensors
        min_measurements = min(len(values) for values in sensor_data.values())
        print(f"Using {min_measurements} measurements from each sensor")
        
        # Create combined vectors, filling missing sensors with 0
        for i in range(min_measurements):
            combined_vector = []
            for sensor in pn_sensors:
                if sensor in sensor_data:
                    combined_vector.append(sensor_data[sensor][i])
                else:
                    combined_vector.append(0)  # Fill missing sensors with 0
            X_array.append(combined_vector)
            analyte_labels.append(analyte)
            conc_labels.append(conc)

# Convert to numpy array and check shape
X_array = np.array(X_array)
if len(X_array) == 0:
    raise ValueError("No valid data combinations found for PCA analysis")

print(f"\nFinal data matrix shape: {X_array.shape}")
print("Vector components represent:", pn_sensors)

# Perform PCA
print("\nPerforming PCA...")
n_components = min(2, X_array.shape[0], X_array.shape[1])  # Ensure we don't request more components than possible
if n_components < 2:
    print(f"Warning: Can only compute {n_components} component(s) due to data limitations")
    print(f"Data shape: {X_array.shape} (samples × features)")
    
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(X_array)

# If we only got one component, add a zero second component for 2D plotting
if n_components == 1:
    pca_result = np.column_stack([pca_result, np.zeros_like(pca_result)])
    explained_variance = np.append(pca.explained_variance_ratio_ * 100, 0)
else:
    explained_variance = pca.explained_variance_ratio_ * 100

print(f"\nExplained variance:")
for i in range(len(explained_variance)):
    print(f"PC{i+1}: {explained_variance[i]:.1f}%")

# Set plotting parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

colors = {
    'IPA': '#1f77b4',    # blue
    'EtOH': '#ff7f0e',   # orange
    'Ace': '#2ca02c'     # green
}

markers = {}
for conc in sorted(clean_data['Conc'].unique()):
    markers[conc] = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D'][len(markers) % 9]

# Figure 1: All individual measurements
plt.figure(figsize=(12, 6), dpi=100)

for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        mask = (np.array(analyte_labels) == analyte) & (np.array(conc_labels) == conc)
        if np.any(mask):
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       c=colors.get(analyte, 'gray'),
                       marker=markers.get(conc, 'o'),
                       s=100,
                       alpha=0.7,
                       label=f'{analyte} {conc}ppm')

plt.grid(True, linestyle='-', alpha=1.0, color='gray', linewidth=0.5)
plt.tick_params(axis='both', which='major', length=6, width=1, labelsize=10, direction='out')

plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance explained)',
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance explained)',
          fontsize=12, fontweight='bold')
plt.title('PCA - All Measurements\nCombined PN Sensors',
         fontsize=14, fontweight='bold', pad=20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PCA_combined_all_measurements.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

# Figure 2: Mean values
plt.figure(figsize=(12, 6), dpi=100)

for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        mask = (np.array(analyte_labels) == analyte) & (np.array(conc_labels) == conc)
        if np.any(mask):
            mean_point = np.mean(pca_result[mask], axis=0)
            plt.scatter(mean_point[0], mean_point[1],
                       c=colors.get(analyte, 'gray'),
                       marker=markers.get(conc, 'o'),
                       s=200,
                       label=f'{analyte} {conc}ppm')

plt.grid(True, linestyle='-', alpha=1.0, color='gray', linewidth=0.5)
plt.tick_params(axis='both', which='major', length=6, width=1, labelsize=10, direction='out')

plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance explained)',
          fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance explained)',
          fontsize=12, fontweight='bold')
plt.title('PCA - Mean Values\nCombined PN Sensors',
         fontsize=14, fontweight='bold', pad=20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PCA_combined_mean_values.png'),
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

# Save results
results = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
results['Analyte'] = analyte_labels
results['Concentration'] = conc_labels
results.to_csv(os.path.join(output_dir, 'PCA_combined_results.csv'), index=False)

# Save loadings with sensor labels
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=pn_sensors
)
loadings.to_csv(os.path.join(output_dir, 'PCA_combined_loadings.csv'))

print("\nAnalysis complete! Check the PCA_results directory for outputs.")