import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os

# Create output directory
output_dir = 'PCA_results'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv("summary_output/summary.csv")
print(f"Loaded {len(data)} measurements")

# After loading data, print initial data distribution
print("\nInitial Data Distribution:")
for analyte in data['Analyte'].unique():
    for conc in data['Conc'].unique():
        count = len(data[(data['Analyte'] == analyte) & (data['Conc'] == conc)])
        print(f"{analyte} at {conc}ppm: {count} total measurements")

# After loading data, add this analysis
print("\nData Summary:")
print(f"Number of unique sensors: {len(data['Sensor ID'].unique())}")
print("\nNumber of measurements for each combination:")
print("Analyte | Concentration | Sensor | Count")
print("-" * 50)
for analyte in data['Analyte'].unique():
    for conc in data['Conc'].unique():
        for sensor in data['Sensor ID'].unique():
            count = len(data[(data['Analyte'] == analyte) & 
                           (data['Conc'] == conc) & 
                           (data['Sensor ID'] == sensor)])
            if count > 0:
                print(f"{analyte:6} | {conc:12} | {sensor:6} | {count:5}")

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    if len(outliers) > 0:
        print(f"\nOutliers found in {column}:")
        print(outliers)
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for each sensor-analyte-concentration combination
clean_data = pd.DataFrame()
for sensor in data['Sensor ID'].unique():
    for analyte in data['Analyte'].unique():
        for conc in data['Conc'].unique():
            subset = data[(data['Sensor ID'] == sensor) & 
                         (data['Analyte'] == analyte) & 
                         (data['Conc'] == conc)].copy()
        if not subset.empty:
                clean_subset = remove_outliers(subset, 'Delta R On')
                clean_data = pd.concat([clean_data, clean_subset])

print(f"\nOriginal data points: {len(data)}")
print(f"Data points after outlier removal: {len(clean_data)}")

# After outlier removal, print cleaned data distribution
print("\nAfter Outlier Removal:")
for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        count = len(clean_data[(clean_data['Analyte'] == analyte) & (clean_data['Conc'] == conc)])
        print(f"{analyte} at {conc}ppm: {count} measurements")

# After outlier removal, add this analysis
print("\nAfter outlier removal:")
print("Analyte | Concentration | Sensor | Count")
print("-" * 50)
for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        for sensor in clean_data['Sensor ID'].unique():
            count = len(clean_data[(clean_data['Analyte'] == analyte) & 
                                 (clean_data['Conc'] == conc) & 
                                 (clean_data['Sensor ID'] == sensor)])
            if count > 0:
                print(f"{analyte:6} | {conc:12} | {sensor:6} | {count:5}")

# Instead of grouping by mean, create a matrix where each row is a complete measurement
print("\nPreparing data matrix...")

# Create a matrix where each row is a complete measurement across all sensors
sensor_matrix = []
analyte_labels = []
conc_labels = []

# Get unique sensor IDs in a consistent order
sensor_ids = sorted(clean_data['Sensor ID'].unique())
print(f"Using sensors: {sensor_ids}")

# Before creating measurement matrix, print detailed sensor counts
print("\nDetailed Sensor Counts per Combination:")
for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        print(f"\n{analyte} at {conc}ppm:")
        subset = clean_data[(clean_data['Analyte'] == analyte) & (clean_data['Conc'] == conc)]
        for sensor in sensor_ids:
            count = len(subset[subset['Sensor ID'] == sensor])
            print(f"  Sensor {sensor}: {count} measurements")

# Modify the measurement matrix creation
sensor_matrix = []
analyte_labels = []
conc_labels = []

# Lower the sensor requirement threshold to ensure more data is included
MIN_SENSORS_REQUIRED = 3  # Adjusted from 6 to 3

for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        subset = clean_data[(clean_data['Analyte'] == analyte) & 
                          (clean_data['Conc'] == conc)]
        
        # Get minimum number of complete measurements across sensors
        min_measurements = float('inf')
        sensor_data = {}
        
        for sensor in sensor_ids:
            measurements = subset[subset['Sensor ID'] == sensor]['Delta R On'].values
            if len(measurements) > 0:  # Only consider sensors with measurements
                sensor_data[sensor] = measurements
                min_measurements = min(min_measurements, len(measurements))
        
        # Relaxed condition for including measurements
        if len(sensor_data) >= MIN_SENSORS_REQUIRED and min_measurements > 0:
            print(f"Processing {analyte} at {conc}ppm with {len(sensor_data)} sensors")
            for i in range(min_measurements):
                row = []
                for sensor in sensor_ids:
                    if sensor in sensor_data:
                        row.append(sensor_data[sensor][i])
                    else:
                        # Use mean value for missing sensors
                        mean_value = np.mean([sensor_data[s][i] for s in sensor_data.keys()])
                        row.append(mean_value)
                sensor_matrix.append(row)
                analyte_labels.append(analyte)
                conc_labels.append(conc)
        else:
            print(f"Skipping {analyte} at {conc}ppm due to insufficient data (only {len(sensor_data)} sensors)")

# Print distribution before PCA
print("\nDistribution before PCA:")
all_analytes = sorted(data['Analyte'].unique())  # Get from original data
all_concs = sorted(data['Conc'].unique())        # Get from original data
for analyte in all_analytes:
    for conc in all_concs:
        count = sum((np.array(analyte_labels) == analyte) & (np.array(conc_labels) == conc))
        print(f"{analyte} at {conc}ppm: {count} measurements")

# Convert to numpy array
X = np.array(sensor_matrix)
print(f"\nFinal data matrix shape: {X.shape}")

# Perform PCA
print("\nPerforming PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_ * 100
print(f"\nExplained variance:")
print(f"PC1: {explained_variance[0]:.1f}%")
print(f"PC2: {explained_variance[1]:.1f}%")

# Update plotting parameters to ensure all analytes are included
colors = {
    'Ace': 'red',
    'EtOH': 'blue',
    'IPA': 'green',
    'MeOH': 'purple',  # Add any additional analytes
    'Water': 'orange'  # Add any additional analytes
}

# Make sure all your concentrations are included
markers = {}
for conc in sorted(data['Conc'].unique()):
    markers[conc] = ['o', 's', '^', 'v', '<', '>', 'p', 'h', '8', 'D'][len(markers) % 10]

# Figure 1: Individual measurements
plt.figure(figsize=(12, 8))  # Increased figure size
for analyte in all_analytes:
    for conc in all_concs:
        mask = (np.array(analyte_labels) == analyte) & (np.array(conc_labels) == conc)
        if np.any(mask):
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=colors.get(analyte, 'gray'),  # Default to gray if analyte not in colors
                       marker=markers.get(conc, 'o'),  # Default to circle if conc not in markers
                       s=marker_size,
                       alpha=0.7,  # Added transparency
                       label=f'{analyte} {conc}ppm')

plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance explained)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance explained)')
plt.title('PCA - Individual Measurements')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PCA_individual_measurements.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Mean values with similar updates
plt.figure(figsize=(12, 8))
X_pca_mean = []
mean_analyte_labels = []
mean_conc_labels = []

# Calculate and plot means
for analyte in all_analytes:
    for conc in all_concs:
        mask = (np.array(analyte_labels) == analyte) & (np.array(conc_labels) == conc)
        if np.any(mask):
            mean_point = np.mean(pca_result[mask], axis=0)
            X_pca_mean.append(mean_point)
            mean_analyte_labels.append(analyte)
            mean_conc_labels.append(conc)
            
            plt.scatter(mean_point[0], mean_point[1],
                       c=colors.get(analyte, 'gray'),
                       marker=markers.get(conc, 'o'),
                       s=marker_size * 1.5,  # Slightly larger markers for means
                       label=f'{analyte} {conc}ppm')

plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance explained)')
plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance explained)')
plt.title('PCA - Mean Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PCA_mean_values.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save mean PCA results
mean_results = pd.DataFrame(X_pca_mean, columns=['PC1', 'PC2'])
mean_results['Analyte'] = mean_analyte_labels
mean_results['Concentration'] = mean_conc_labels
mean_results.to_csv(os.path.join(output_dir, 'PCA_mean_results.csv'), index=False)

# Print loadings
print("\nPCA Loadings:")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=sensor_ids
)
print(loadings)
loadings.to_csv(os.path.join(output_dir, 'PCA_loadings.csv'))

print("\nAnalysis complete! Check the PCA_results directory for outputs.")
