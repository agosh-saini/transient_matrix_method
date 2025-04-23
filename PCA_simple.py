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

# Function to remove extreme outliers (2x IQR instead of 3x)
def remove_extreme_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR  # Changed from 3 to 2
    upper_bound = Q3 + 2 * IQR  # Changed from 3 to 2
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    if len(outliers) > 0:
        print(f"\nExtreme outliers found for {df['Analyte'].iloc[0]} at {df['Conc'].iloc[0]}ppm:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Outlier values: {outliers[column].values}")
        print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Kept {len(clean)} out of {len(df)} measurements")
    
    return clean

# Remove outliers for each sensor-analyte-concentration combination
print("\nRemoving outliers...")
clean_data = pd.DataFrame()
for analyte in data['Analyte'].unique():
    for conc in data['Conc'].unique():
        subset = data[(data['Analyte'] == analyte) & (data['Conc'] == conc)].copy()
        if not subset.empty:
            clean_subset = remove_extreme_outliers(subset, 'Delta R On')
            clean_data = pd.concat([clean_data, clean_subset])

print(f"\nOriginal data points: {len(data)}")
print(f"Data points after outlier removal: {len(clean_data)}")

# Prepare data for PCA - keeping all measurements
print("\nPreparing data for PCA...")
sensor_ids = sorted(clean_data['Sensor ID'].unique())

# Create a matrix where each row is a measurement
X_array = []
analyte_labels = []
conc_labels = []

# Group by Analyte and Concentration
for (analyte, conc), group in clean_data.groupby(['Analyte', 'Conc']):
    # Get all measurements for this combination
    for sensor_id in sensor_ids:
        sensor_data = group[group['Sensor ID'] == sensor_id]['Delta R On'].values
        if len(sensor_data) > 0:
            for value in sensor_data:
                # Create a row for each measurement
                row = []
                for s_id in sensor_ids:
                    if s_id == sensor_id:
                        row.append(value)
                    else:
                        # Get a random measurement from the same analyte/conc for other sensors
                        other_data = group[group['Sensor ID'] == s_id]['Delta R On'].values
                        if len(other_data) > 0:
                            row.append(np.random.choice(other_data))
                        else:
                            row.append(0)
                X_array.append(row)
                analyte_labels.append(analyte)
                conc_labels.append(conc)

X_array = np.array(X_array)

# Perform PCA
print("\nPerforming PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_array)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_ * 100
print(f"\nExplained variance:")
print(f"PC1: {explained_variance[0]:.1f}%")
print(f"PC2: {explained_variance[1]:.1f}%")

# Set global font settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# First, print unique analytes to see exact names
print("\nUnique analytes in data:", clean_data['Analyte'].unique())

# Updated colors with exact matching of analyte names
colors = {
    'IPA': '#1f77b4',    # blue
    'EtOH': '#ff7f0e',   # orange  # Note: might be 'ETOH' or 'EtOH' in your data
    'Ace': '#2ca02c'     # green   # Note: might be 'ACE' or 'Ace' in your data
}

# Add debug print to check color mapping
for analyte in clean_data['Analyte'].unique():
    print(f"Analyte: {analyte}, Color: {colors.get(analyte, 'gray')}")

markers = {}
for conc in sorted(clean_data['Conc'].unique()):
    markers[conc] = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D'][len(markers) % 9]

# Figure 1: All individual measurements
plt.figure(figsize=(12, 6), dpi=100)

for analyte in clean_data['Analyte'].unique():
    for conc in clean_data['Conc'].unique():
        mask = (np.array(analyte_labels) == analyte) & (np.array(conc_labels) == conc)
        if np.any(mask):
            print(f"Plotting {analyte} with color {colors.get(analyte, 'gray')}")  # Debug print
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                       c=colors.get(analyte, 'gray'),
                       marker=markers.get(conc, 'o'),
                       s=100,
                       alpha=0.7,
                       label=f'{analyte} {conc}ppm')

# Style the grid
plt.grid(True, 
         linestyle='-', 
         alpha=1.0,
         color='gray',
         linewidth=0.5)

# Style the ticks
plt.tick_params(axis='both',
                which='major',
                length=6,
                width=1,
                labelsize=10,
                direction='out')

# Style labels and title
plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance explained)',
          fontsize=12,
          fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance explained)',
          fontsize=12,
          fontweight='bold')
plt.title('PCA - All Individual Measurements\n(Extreme Outliers Removed)',
         fontsize=14,
         fontweight='bold',
         pad=20)

# Style legend
plt.legend(bbox_to_anchor=(1.05, 1),
          loc='upper left',
          frameon=True,
          fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PCA_all_measurements_no_outliers.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()

# Figure 2: Mean values
plt.figure(figsize=(12, 6), dpi=100)  # Changed to 12x6 inches

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

# Style the grid
plt.grid(True, 
         linestyle='-', 
         alpha=1.0,
         color='gray',
         linewidth=0.5)

# Style the ticks
plt.tick_params(axis='both',
                which='major',
                length=6,
                width=1,
                labelsize=10,
                direction='out')

# Style labels and title
plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance explained)',
          fontsize=12,
          fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance explained)',
          fontsize=12,
          fontweight='bold')
plt.title('PCA - Mean Values\n(Extreme Outliers Removed)',
         fontsize=14,
         fontweight='bold',
         pad=20)

# Style legend
plt.legend(bbox_to_anchor=(1.05, 1),
          loc='upper left',
          frameon=True,
          fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PCA_mean_values_no_outliers.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()

# Save PCA results
results = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
results['Analyte'] = analyte_labels
results['Concentration'] = conc_labels
results.to_csv(os.path.join(output_dir, 'PCA_results_no_outliers.csv'), index=False)

# Save PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=sensor_ids
)
loadings.to_csv(os.path.join(output_dir, 'PCA_loadings_no_outliers.csv'))

print("\nAnalysis complete! Check the PCA_results directory for outputs.") 