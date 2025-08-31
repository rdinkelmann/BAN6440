# Importing the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1 is to download the NYC taxi file from AWS and import and it into python
Data_Assignment4 = r"C:\Users\dinkelmann\Desktop\Nexford\BAN6440\Module 4\yellow_tripdata_2023-01.parquet"
# Formatting the Parquet file into a dataframe to be used
df = pd.read_parquet(Data_Assignment4)

# Step 2 is to perform the data processing and reference the columns that will be used for the k-means analysis
columns = ['trip_distance', 'fare_amount']  # Only the most meaningful features
# Need to ensure that the data is clean with no duplicates, missing values, or unrealistic values
X = df[columns].fillna(0)
X = X.drop_duplicates()

# After reviewing the data, outliers have been detected and will be filtered out of the dataframe
X = X[(X['trip_distance'] > 0) &
      (X['fare_amount'] >= 0) &
      (X['trip_distance'] < 100)]  # Trips further than 100 miles are outliers

# Step 3 is to standardize the data to ensure that all the attributes are equally weighted against each other
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Using the elbow method to determine the best amount of clusters
wcss = []
k_values = range(1, 11)

# Testing out the different k-values
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=28)
    kmeans.fit(X_sc)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method to select the optimal number of clusters
plt.figure(figsize=(8,6))
plt.plot(k_values, wcss, 'bo-', markersize=8)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Step 4 is to apply the K-Means clustering algorithm, based on the elbow method, the inflection point is at 4 clusters
k_means = KMeans(n_clusters=4, random_state=28)
k_means.fit(X_sc)

# Step 5 is to add the cluster labels back to the original dataframe to see which trip fit to which cluster
X['cluster'] = k_means.labels_

# Step 6 is to examine the cluster centers points
cluster_cent = scaler.inverse_transform(k_means.cluster_centers_)
cluster_df = pd.DataFrame(cluster_cent, columns=columns)
print("Cluster Center Points")
print(cluster_df)

# Step 7 is to plot the 3 clusters using the fare amount and trip distance
plt.figure(figsize=(8,6))
plt.scatter(X['trip_distance'], X['fare_amount'], c=X['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Trip Distance')
plt.ylabel('Fare Amount')
plt.title('K-Means Clusters Analysis of NYC Taxi Trips')
plt.show()

# Step 8 is to check the average passenger count in each cluster
passenger_summary = df.loc[X.index].groupby(X['cluster'])['passenger_count'].mean()
print("Avg passenger count per cluster")
print(passenger_summary)
