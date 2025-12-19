# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

# Step 2: Load dataset
df = pd.read_csv("Mall_Customers (2).csv")
print("First 5 rows of dataset:")
print(df.head())

# Step 3: Preprocess data
df.drop("CustomerID", axis=1, inplace=True)
df["Genre"] = LabelEncoder().fit_transform(df["Genre"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 4: Plot dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(linkage(X_scaled, method="ward"))
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 5: Apply Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
labels = hc.fit_predict(X_scaled)
df["Cluster"] = labels
print("\nDataset with Cluster Labels:")
print(df.head())

# Step 6: Visualize clusters
plt.figure(figsize=(8, 5))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using Hierarchical Clustering")
plt.show()

# Step 7: Evaluate clustering
sil_score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", sil_score)

# Step 8: Save model and scaler
joblib.dump(hc, "hierarchical_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and Scaler saved successfully!")
