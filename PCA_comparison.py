import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import os

############# Configurations ############
seed = 20785161
np.random.seed(seed)
output_folder = "combo"
os.makedirs(output_folder, exist_ok=True)

############ Functions ############
def is_valid_combination(combo):
    sensors = [sensor.split('.')[0] for sensor in combo]
    return len(set(sensors)) == len(sensors)

# Two analytes: IPA and EtOH
analyte_values = {
    "IPA": 1,
    "EtOH": 2,
    "Ace": 3
}

# Concentrations mapped to classes
class_values = {
    250: 1,
    350: 2,
    450: 3
}

inv_analyte_values = {v: k for k, v in analyte_values.items()}
inv_class_values = {v: k for k, v in class_values.items()}

############# Load Data ############
data = pd.read_csv("summary_output/summary.csv")

# Debug: Check the data after loading
print("Data Sample:\n", data.head())
print("Data Info:\n", data.info())

############# Create a Dictionary for Data Organization ############
data_dict = {}

# Iterate through unique combinations of analyte and concentration
for analyte in data["Analyte"].unique():
    for conc in data["Conc"].unique():
        subset = data[(data["Analyte"] == analyte) & (data["Conc"] == conc)]
        if not subset.empty:
            sensors_data = {}
            for sensor_id in subset["Sensor ID"].unique():
                sensor_subset = subset[subset["Sensor ID"] == sensor_id]
                sensors_data[sensor_id] = {
                    "DRon": sensor_subset["Delta R On"].values,
                    "DRoff": sensor_subset["Delta R Off"].values,
                    #"TauOn": np.log(sensor_subset["Tau On"].values + 1e-9),
                    #"TauOff": np.log(sensor_subset["Tau Off"].values + 1e-9)
                }
            data_dict[(analyte, conc)] = sensors_data

############# Create Final DataFrame ############
final_rows = []

for (analyte, conc), sensors in data_dict.items():
    for i in range(len(next(iter(sensors.values()))["DRon"])):  # Assuming all sensors have the same number of replicates
        row = {
            "Analyte": analyte,
            "Conc": conc
        }
        for sensor_id, values in sensors.items():
            row[f"{sensor_id}-DRon"] = values["DRon"][i]
            row[f"{sensor_id}-DRoff"] = values["DRoff"][i]
            #row[f"{sensor_id}-TauOn"] = values["TauOn"][i]
            #row[f"{sensor_id}-TauOff"] = values["TauOff"][i]
        final_rows.append(row)

final_data = pd.DataFrame(final_rows)

# Check if final_data is empty
if final_data.empty:
    print("Final DataFrame is empty! Check input data and aggregation.")
else:
    print("Final DataFrame Sample:\n", final_data.head())

# Save combined DataFrame
combined_df_path = os.path.join(output_folder, "final_combined_dataframe.csv")
final_data.to_csv(combined_df_path, index=False)
print(f"Final combined dataframe saved to {combined_df_path}")

############# Perform PCA Analysis for Incremental Sensor Combinations #############
sensor_columns = [col for col in final_data.columns if re.match(r"PN\d+\.\d+-", col)]

# Debugging: Print available sensor columns
print("Sensor columns detected:", sensor_columns)

sensor_base_ids = sorted(set(re.match(r"(PN\d+)\.\d+", col).group(1) for col in sensor_columns if re.match(r"(PN\d+)\.\d+", col)))
print("Sensor base IDs:", sensor_base_ids)

results = []

# Iterate through incremental combinations of sensors
for num_sensors in range(1, len(sensor_base_ids) + 1):
    print(f"Performing PCA for {num_sensors} sensor(s)...")
    sensor_combinations = combinations(sensor_base_ids, num_sensors)
    
    for sensor_combo in sensor_combinations:
        # Collect all features for selected sensors
        selected_features = [col for col in sensor_columns if any(col.startswith(f"{sensor_id}.") for sensor_id in sensor_combo)]
        if not selected_features:
            print(f"No features found for sensor combination: {sensor_combo}")
            continue

        # Drop rows with missing values only for the selected features
        X = final_data[selected_features].dropna()
        y_conc = final_data.loc[X.index, "Conc"].values
        y_analyte = final_data.loc[X.index, "Analyte"].values
        y_combined = [f"{analyte}-{conc}" for analyte, conc in zip(y_analyte, y_conc)]

        print(f"Sensor combination {sensor_combo} - Data shape: {X.shape}")

        if X.shape[0] < 2 or X.shape[1] < 2:
            print(f"Skipping {sensor_combo}: insufficient data for PCA.")
            continue

        # Apply PCA
        n_components = min(2, X.shape[1])
        pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
        X_pca = pca.fit_transform(X)
        pca_results = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

        ########### Analyte Classification ###########
        print("Running analyte classification...")
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y_analyte, test_size=0.3, random_state=seed)
        classifier_analyte = SVC(kernel="linear", random_state=seed)
        classifier_analyte.fit(X_train, y_train)
        y_pred_analyte = classifier_analyte.predict(X_test)
        accuracy_analyte = accuracy_score(y_test, y_pred_analyte)

        ########### Combined Analyte + Concentration Classification ###########
        print("Running analyte + concentration classification...")
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y_combined, test_size=0.3, random_state=seed)
        classifier_combined = SVC(kernel="linear", random_state=seed)
        classifier_combined.fit(X_train, y_train)
        y_pred_combined = classifier_combined.predict(X_test)
        accuracy_combined = accuracy_score(y_test, y_pred_combined)

        ########### Results and Metrics ###########
        silhouette_avg = silhouette_score(X_pca, y_conc)
        db_index = davies_bouldin_score(X_pca, y_conc)
        
        # Save results
        results.append({
            "Num Sensors": num_sensors,
            "Sensors": sensor_combo,
            "Explained Variance Ratio": pca.explained_variance_ratio_.tolist(),
            "Total Variance Explained": pca.explained_variance_ratio_.sum(),
            "Silhouette Score": silhouette_avg,
            "Davies-Bouldin Index": db_index,
            "Accuracy - Analyte": accuracy_analyte,
            "Accuracy - Combined": accuracy_combined
        })

        # Visualization for PCA class separation
        plt.figure(figsize=(10, 8))
        pca_results["Analyte"] = y_analyte
        pca_results["Conc"] = y_conc
        sns.scatterplot(data=pca_results, x="PC1", y="PC2", hue="Analyte", style="Conc", palette="viridis")
        plt.title(f"PCA Analysis - {num_sensors} Sensors")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"pca_class_separation_{sensor_combo}_analyte.png"))
        plt.close()

# Save Results
results_df = pd.DataFrame(results)
results_path = os.path.join(output_folder, "pca_analyte_conc_results.csv")
results_df.to_csv(results_path, index=False)
print(f"PCA results with analyte and combined classification saved to {results_path}")
