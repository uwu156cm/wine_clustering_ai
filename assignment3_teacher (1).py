#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Wine dataset
df = pd.read_csv("Wine.csv")
X = df.drop(columns=['Customer_Segment', 'Proline', 'Magnesium'])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=0)
kmeans.fit(X)
cluster_assignments = kmeans.labels_
df['Cluster'] = cluster_assignments

# Train a Perceptron model
X_train, X_test, y_train, y_test = train_test_split(X, cluster_assignments, test_size=0.2, random_state=42)
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# Create a Streamlit web app
st.title("Wine Clustering and Perceptron Model")

st.write("### Cluster Assignments:")
st.dataframe(df)

st.write("### Perceptron Model Accuracy:")
y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.write("### Predict New Data:")
st.write("Enter the features for a new data point:")
new_data = st.text_input("Enter data as comma-separated values (e.g., 13.10,1.70,2.50,12.5,2.80,3.20,0.25,2.00,6.15,0.98,3.50):")
if new_data:
    new_data = np.array([float(x) for x in new_data.split(',')]).reshape(1, -1)
    pred = perceptron.predict(new_data)
    st.write(f"Predicted Cluster: {pred[0]}")


# In[ ]:




