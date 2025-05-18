from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
file_path = r'C:\Users\naina\Downloads\Resume Project\ML Based Coral Water Analysis and Optimization\Clustering Algorithm\K Means Clustering\unsuitable.csv'
unsuitable_df = pd.read_csv(file_path)
columns_to_remove = ['Suitability', 'Predicted_Suitability']
unsuitable_features = unsuitable_df.drop(columns=columns_to_remove)
scaler = StandardScaler()
unsuitable_scaled = scaler.fit_transform(unsuitable_features)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
unsuitable_df['Cluster'] = kmeans.fit_predict(unsuitable_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=unsuitable_features.columns)
if centroid_df.iloc[0].mean() > centroid_df.iloc[1].mean():
    cluster_labels = {0: 'Far Suitable', 1: 'Near Suitable'}
else:
    cluster_labels = {0: 'Near Suitable', 1: 'Far Suitable'}
unsuitable_df['Cluster_Label'] = unsuitable_df['Cluster'].map(cluster_labels)
near_suitable_df = unsuitable_df[unsuitable_df['Cluster_Label'] == 'Near Suitable']
far_suitable_df = unsuitable_df[unsuitable_df['Cluster_Label'] == 'Far Suitable']
near_suitable_df.to_csv('near_suitable_data.csv', index=False)
far_suitable_df.to_csv('far_suitable_data.csv', index=False)
print('Clustered datasets saved as near_suitable_data.csv and far_suitable_data.csv.')
silhouette = silhouette_score(unsuitable_scaled, unsuitable_df['Cluster'])
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(unsuitable_scaled)
plt.figure(figsize=(8, 6))
colors = {'Near Suitable': 'blue', 'Far Suitable': 'red'}
for label, color in colors.items():
    subset = unsuitable_df[unsuitable_df['Cluster_Label'] == label]
    plt.scatter(reduced_data[subset.index, 0], reduced_data[subset.index, 1],
                label=f'{label} (Count: {len(subset)})', color=color, alpha=0.6)
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
cluster_counts = unsuitable_df['Cluster_Label'].value_counts()
print('Silhouette Score:', silhouette)
print('\nCluster Distribution:\n', cluster_counts)
