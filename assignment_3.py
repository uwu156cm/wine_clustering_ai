#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the dataset
data = pd.read_csv('Wine.csv')


# In[10]:


# Feature Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(['Customer_Segment'], axis=1))

# Clustering
num_clusters = st.sidebar.slider("Select the number of clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
data['Cluster'] = kmeans.fit_predict(data_scaled)


# In[16]:


# Visualization
st.title("Wine Clustering")
st.sidebar.subheader("Clustering Parameters")
st.sidebar.write("Number of Clusters:", num_clusters)

# Create a scatter plot of two features (you can choose any two)
x_axis = st.sidebar.selectbox("Select X-Axis Feature", data.columns[:-1])
y_axis = st.sidebar.selectbox("Select Y-Axis Feature", data.columns[:-1])

# Scatter plot with clusters
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x=x_axis, y=y_axis, hue='Cluster', palette='viridis', s=100, ax=ax)
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title(f'Clustering based on {x_axis} vs {y_axis}')
st.pyplot(fig)


# In[17]:


# Streamlit App
st.write("### Data Table")
st.write(data)  # Display the data table with cluster assignments


# In[ ]:




