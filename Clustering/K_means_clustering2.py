import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

cust_df = cust_df.drop('Address', axis=1)

# Drop NaNs from the dataframe
cust_df = cust_df.dropna()
print(cust_df)
#cust_df.info()

#Standadize dataset
X = cust_df.values[:,1:] # leaves out `Customer ID`
Clus_dataSet = StandardScaler().fit_transform(X)

# Cluster with k=3
clusterNum = 3
k_means = KMeans(n_clusters=clusterNum, init='k-means++', n_init=12)
k_means.fit(Clus_dataSet)

# Extract cluster labels
labels = k_means.labels_

cust_df["Clus_km"] = labels
cust_df.groupby('Clus_km').mean()

area = np.pi * (cust_df['Edu'])**2
plt.scatter(cust_df['Age'], cust_df['Income'], s=area, c=labels.astype(float), cmap='tab10', ec='k', alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()


# Create interactive 3D scatter plot
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Edu'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

fig.show()

fig = px.scatter_3d(cust_df, x='Edu', y='Age', z='Income', opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
    xaxis=dict(title='Edu'),
    yaxis=dict(title='Age'),
    zaxis=dict(title='Income')
))
fig.show()

