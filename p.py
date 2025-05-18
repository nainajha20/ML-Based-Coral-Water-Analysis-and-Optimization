import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=FutureWarning)
file_path = r'C:\Users\naina\Downloads\Resume Project\ML Based Coral Water Analysis and Optimization\Classification of Dataset using AdaBoost\Expanded_Water_Quality_Dataset.csv'
data = pd.read_csv(file_path)
print("Column Names:", data.columns)
feature_weights = {
    'Dissolved Oxygen (mg/L)': 5.0,
    'pH': 3.5,
    'Temperature (°C)': -0.4,
    'Ammonia Nitrogen (mg/L)': -4.3,
    'Phosphate (mg/L)': -3.6,
    'Nitrate (mg/L)': -3.0,
    'Turbidity (NTU)': -2.6,
    'Biological Oxygen Demand (BOD) (mg/L)': -3.5
}
selected_features = [col for col in feature_weights.keys() if col in data.columns]
data['Quality_Index'] = sum(data[feature] * weight for feature, weight in feature_weights.items() if feature in data.columns)
print("Quality Index Stats:")
print(data['Quality_Index'].describe())
print("Minimum Quality Index:", data['Quality_Index'].min())
print("Maximum Quality Index:", data['Quality_Index'].max())
sorted_quality = data['Quality_Index'].sort_values()
suitable_range = (sorted_quality.iloc[70], sorted_quality.iloc[250])
print("Updated Suitable Range:", suitable_range)
data['Suitability'] = data['Quality_Index'].apply(lambda x: 1 if suitable_range[0] <= x <= suitable_range[1] else 0)
print("Number of Suitable Samples:", sum(data['Suitability']))
X = data[selected_features]
y = data['Suitability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
base_classifier = DecisionTreeClassifier(max_depth=1)
clf = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
predictions = clf.predict(X)
data['Predicted_Suitability'] = predictions
suitable = data[data['Predicted_Suitability'] == 1]
unsuitable = data[data['Predicted_Suitability'] == 0]
suitable.to_csv('suitable.csv', index=False)
unsuitable.to_csv('unsuitable.csv', index=False)


