import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

############# Configurations ############
seed = 20785161
np.random.seed(seed)

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

# Ensure Sample ID exists and handle correctly
data = data.rename(columns={"Filename": "Sample ID"})

# Filter to only IPA and EtOH
data = data[data["Analyte"].isin(analyte_values.keys())]

data["AnalyteClass"] = data["Analyte"].map(analyte_values)
data["ConcClass"] = data["Conc"].map(class_values)

pivoted_data = data.pivot(index="Sample ID", columns="Sensor ID", values=["Delta R On", "Delta R Off"])
pivoted_data = pivoted_data.reset_index()
pivoted_data["AnalyteClass"] = data.groupby("Sample ID")["AnalyteClass"].first().values
pivoted_data["ConcClass"] = data.groupby("Sample ID")["ConcClass"].first().values

pivoted_data = pivoted_data.fillna(0)

############# Train-Test Split ############
X = pivoted_data.drop(columns=["Sample ID", "ConcClass", "AnalyteClass"])
y_analyte = pivoted_data["AnalyteClass"]
y_conc = pivoted_data["ConcClass"]

X_train, X_test, y_train_analyte, y_test_analyte, y_train_conc, y_test_conc = train_test_split(
    X, y_analyte, y_conc, test_size=0.3, random_state=seed, stratify=y_analyte
)

############# Analyte Classification (SVC) ############
scaler_analyte = StandardScaler()
X_train_scaled = scaler_analyte.fit_transform(X_train)
X_test_scaled = scaler_analyte.transform(X_test)

analyte_model_svc = SVC(class_weight='balanced', kernel='rbf', C=100, gamma='auto', random_state=seed)
analyte_model_svc.fit(X_train_scaled, y_train_analyte)
y_pred_analyte_svc = analyte_model_svc.predict(X_test_scaled)

analyte_accuracy_svc = accuracy_score(y_test_analyte, y_pred_analyte_svc)
print(f"Analyte Classification Accuracy (SVC): {analyte_accuracy_svc:.2f}")

cm_analyte_svc = confusion_matrix(y_test_analyte, y_pred_analyte_svc)
print("Confusion Matrix (SVC - Analyte):")
print(cm_analyte_svc)

############# Analyte Classification (KNN) ############
analyte_model_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
analyte_model_knn.fit(X_train_scaled, y_train_analyte)
y_pred_analyte_knn = analyte_model_knn.predict(X_test_scaled)

analyte_accuracy_knn = accuracy_score(y_test_analyte, y_pred_analyte_knn)
print(f"Analyte Classification Accuracy (KNN): {analyte_accuracy_knn:.2f}")

cm_analyte_knn = confusion_matrix(y_test_analyte, y_pred_analyte_knn)
print("Confusion Matrix (KNN - Analyte):")
print(cm_analyte_knn)

############# Separate Concentration Classification per Analyte ############
# Use train-test split already performed
for model_name, analyte_model, y_pred_analyte in zip([
    "SVC", "KNN"], [analyte_model_svc, analyte_model_knn], [y_pred_analyte_svc, y_pred_analyte_knn]
):
    print(f"\n=== Concentration Classification ({model_name}) ===")
    # IPA
    X_train_ipa = X_train[y_train_analyte == analyte_values["IPA"]]
    y_train_ipa = y_train_conc[y_train_analyte == analyte_values["IPA"]]

    X_test_ipa = X_test[y_test_analyte == analyte_values["IPA"]]
    y_test_ipa = y_test_conc[y_test_analyte == analyte_values["IPA"]]

    scaler_ipa = StandardScaler()
    X_train_ipa_scaled = scaler_ipa.fit_transform(X_train_ipa)
    X_test_ipa_scaled = scaler_ipa.transform(X_test_ipa)

    ipa_model = KNeighborsClassifier(n_neighbors=5) if model_name == "KNN" else SVC(
        class_weight='balanced', kernel='rbf', C=100, gamma='auto', random_state=seed
    )
    ipa_model.fit(X_train_ipa_scaled, y_train_ipa)
    y_pred_ipa = ipa_model.predict(X_test_ipa_scaled)

    ipa_accuracy = accuracy_score(y_test_ipa, y_pred_ipa)
    print(f"IPA Concentration Classification Accuracy: {ipa_accuracy:.2f}")

    cm_ipa = confusion_matrix(y_test_ipa, y_pred_ipa)
    print(f"Confusion Matrix ({model_name} - IPA):")
    print(cm_ipa)

    # EtOH
    X_train_etoh = X_train[y_train_analyte == analyte_values["EtOH"]]
    y_train_etoh = y_train_conc[y_train_analyte == analyte_values["EtOH"]]

    X_test_etoh = X_test[y_test_analyte == analyte_values["EtOH"]]
    y_test_etoh = y_test_conc[y_test_analyte == analyte_values["EtOH"]]

    scaler_etoh = StandardScaler()
    X_train_etoh_scaled = scaler_etoh.fit_transform(X_train_etoh)
    X_test_etoh_scaled = scaler_etoh.transform(X_test_etoh)

    etoh_model = KNeighborsClassifier(n_neighbors=5) if model_name == "KNN" else SVC(
        class_weight='balanced', kernel='rbf', C=100, gamma='auto', random_state=seed
    )
    etoh_model.fit(X_train_etoh_scaled, y_train_etoh)
    y_pred_etoh = etoh_model.predict(X_test_etoh_scaled)

    etoh_accuracy = accuracy_score(y_test_etoh, y_pred_etoh)
    print(f"EtOH Concentration Classification Accuracy: {etoh_accuracy:.2f}")

    cm_etoh = confusion_matrix(y_test_etoh, y_pred_etoh)
    print(f"Confusion Matrix ({model_name} - EtOH):")
    print(cm_etoh)
