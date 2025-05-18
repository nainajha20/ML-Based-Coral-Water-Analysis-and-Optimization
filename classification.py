# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = r'C:\Users\naina\Downloads\Resume Project\ML Based Coral Water Analysis and Optimization\A Comparative Analysis Using Machine Learning\DATASET.csv'
data = pd.read_csv(file_path)

# Define thresholds for classifying water quality
pH_range = (7.9, 8.4)
dissolved_oxygen_min = 6.0
temperature_range = (23, 28)

# Add 'Suitability' column
data['Suitability'] = data.apply(lambda row: 'suitable' if (
    pH_range[0] <= row['pH'] <= pH_range[1] and
    row['Dissolved Oxygen (mg/L)'] >= dissolved_oxygen_min and
    temperature_range[0] <= row['Temperature (°C)'] <= temperature_range[1]
) else 'unsuitable', axis=1)

# Encode labels (suitable: 1, unsuitable: 0)
label_encoder = LabelEncoder()
data['Suitability'] = label_encoder.fit_transform(data['Suitability'])

# Define features and target
X = data.drop(columns=["Suitability"])
y = data["Suitability"]

# Stratified Split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=6, random_state=42, stratify=y)

# Dictionary to store results
results = {}

# 1. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
results["Decision Tree"] = classification_report(y_test, y_pred_dt, zero_division=0)

# 2. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
results["Random Forest"] = classification_report(y_test, y_pred_rf, zero_division=0)

# 3. SVM
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
results["SVM"] = classification_report(y_test, y_pred_svm, zero_division=0)

# 4. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
results["Logistic Regression"] = classification_report(y_test, y_pred_lr, zero_division=0)

# 5. K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
results["K-Nearest Neighbors"] = classification_report(y_test, y_pred_knn, zero_division=0)

# 6. Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
results["Naive Bayes"] = classification_report(y_test, y_pred_nb, zero_division=0)

# 7. XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results["XGBoost"] = classification_report(y_test, y_pred_xgb, zero_division=0)

# 8. AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
results["AdaBoost"] = classification_report(y_test, y_pred_ada, zero_division=0)

# 9. Artificial Neural Network
from sklearn.neural_network import MLPClassifier
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)
y_pred_ann = ann_model.predict(X_test)
results["ANN"] = classification_report(y_test, y_pred_ann, zero_division=0)

# 10. Extra Trees
from sklearn.ensemble import ExtraTreesClassifier
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)
y_pred_et = et_model.predict(X_test)
results["Extra Trees"] = classification_report(y_test, y_pred_et, zero_division=0)

# Display results
for model_name, report in results.items():
    print(f"Results for {model_name}:\n{report}\n")
