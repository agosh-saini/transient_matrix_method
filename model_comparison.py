######################################################
# Author: Agosh Saini
# Contact: contact@agoshsaini.com
######################################################
# Description: SVC, RFC, and GBC-based classification with feature scaling and model comparison.
######################################################

############# Import ############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

############# Init Value ############

seed = 20785161
np.random.seed(seed)

class_values = dict(zip([900, 1900, 2900], [1, 2, 3]))  # Map concentrations to classes
values = [1, 2, 3]  # Possible class labels

############# Load Data ############

data = pd.read_csv("summary_output/summary.csv")  

# Map concentration values to class labels
data["Conc"] = data["Conc"].map(class_values)

############# Prepare Features ############

# Create a unique identifier for each sample - File name is unique for each sample and can also be used
data['Sample ID'] = data['Filename']

# Pivot the dataset to aggregate sensor data
pivoted_data = data.pivot(index="Sample ID", columns="Sensor ID")[["Delta R On", "Delta R Off"]].reset_index()

# Add the class labels for each sample
pivoted_data["Conc"] = data.groupby("Sample ID")["Conc"].first().values

# Fill missing values with 0 (or a suitable constant)
pivoted_data = pivoted_data.fillna(0)

############# Split Data into Training and Testing Sets ############

X = pivoted_data.drop(columns=["Sample ID", "Conc"])  # Features (sensor data)
y = pivoted_data["Conc"]  # Labels (classes)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

############# Feature Scaling ############

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############# Model Training and Evaluation ############

# Initialize models

models = {
    "SVC": SVC(class_weight='balanced', kernel='rbf', C=100, gamma='auto', random_state=seed),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=seed)
}

# Train and evaluate models
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train the model
    if model_name == "SVC":
        # Use scaled data for SVC
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Use raw data for tree-based models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Store results
    results[model_name] = {
        "Classification Report": classification_report(y_test, y_pred, target_names=[str(v) for v in values]),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred)
    }

############# Display Results ############

for model_name, result in results.items():
    print(f"\n=== {model_name} ===")
    print("Classification Report:")
    print(result["Classification Report"])
    print("Confusion Matrix:")
    print(result["Confusion Matrix"])
    print(f"Accuracy: {result['Accuracy']:.2f}")

############# Plot Confusion Matrix ############
fig, ax = plt.subplots(1, len(results), figsize=(15, 5))  

for i, (model_name, result) in enumerate(results.items()):
    # Extract confusion matrix
    cm = result["Confusion Matrix"]
    
    # Display the confusion matrix with colors
    cax = ax[i].matshow(cm, cmap='Reds')
    ax[i].set_title(f"{model_name} (Accuracy: {result['Accuracy']:.2f})")
    ax[i].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")
    
    # Annotate the matrix with values
    for (row, col), value in np.ndenumerate(cm):
        ax[i].text(col, row, f"{value}", ha='center', va='center', color="black")
    
fig.colorbar(cax, ax=ax[2], fraction=0.046, pad=0.04)

# Adjust layout
plt.tight_layout()
plt.show()