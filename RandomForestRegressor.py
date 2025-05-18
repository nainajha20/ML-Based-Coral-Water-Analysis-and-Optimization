import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
def identify_outliers(df, thresholds):
    outliers = {}
    for feature, (low, high) in thresholds.items():
        if feature in df.columns:
            outliers[feature] = df[(df[feature] < low) | (df[feature] > high)].shape[0]
    return outliers
def adjust_values(df, thresholds):
    for feature, (low, high) in thresholds.items():
        if feature in df.columns:
            df[feature] = df[feature].clip(lower=low, upper=high)
            df[feature] = MinMaxScaler(feature_range=(low, high)).fit_transform(df[[feature]])
    return df
def apply_kmeans(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=['number'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster_Label'] = kmeans.fit_predict(numeric_df)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_df.columns)
    return df, centroids
def apply_random_forest_model(df, suitable_df):
    if "Cluster_Label" not in suitable_df.columns:
        suitable_df, _ = apply_kmeans(suitable_df)
    common_features = list(set(df.columns) & set(suitable_df.columns) - {"Cluster_Label"})
    X = suitable_df[common_features]
    y = suitable_df["Cluster_Label"]
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    df["Predicted_Label"] = model.predict(df[common_features])
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"Random Forest Regression Performance: R² Score = {r2:.4f}, MAE = {mae:.4f}")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y, y=y_pred, color='blue', alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Cluster Labels")
    plt.ylabel("Predicted Cluster Labels")
    plt.title("Random Forest Regression: Actual vs Predicted")
    plt.show()
    return df
if __name__ == "__main__":
    file_path1 = r'C:\Users\naina\Downloads\Resume Project\ML Based Coral Water Analysis and Optimization\Regression Algorithm\Random Forest Regressor\near_suitable_data.csv'
    far_unsuitable_df = pd.read_csv(file_path1)
    suitable_thresholds = {
        'Dissolved Oxygen (mg/L)': (5.0, 9.0),
        'pH': (6.5, 8.5),
        'Temperature (°C)': (22.0, 30.0),
        'Ammonia Nitrogen (mg/L)': (0.0, 1.5),
        'Phosphate (mg/L)': (0.0, 0.1),
        'Nitrate (mg/L)': (0.0, 10.0),
        'Turbidity (NTU)': (0.0, 5.0),
        'Biological Oxygen Demand (BOD) (mg/L)': (0.0, 3.0)
    }
    far_unsuitable_adjusted = adjust_values(far_unsuitable_df.copy(), suitable_thresholds)
    far_unsuitable_clustered, centroids = apply_kmeans(far_unsuitable_adjusted)
    file_path2 = r'C:\Users\naina\Downloads\Resume Project\ML Based Coral Water Analysis and Optimization\Regression Algorithm\Random Forest Regressor\suitable.csv'
    suitable_df = pd.read_csv(file_path2)
    far_unsuitable_adjusted = apply_random_forest_model(far_unsuitable_adjusted, suitable_df)
    far_unsuitable_adjusted.to_csv("transformed_suitable_data.csv", index=False)
    print("Data transformation complete. Saved as 'transformed_suitable_data.csv'.")
