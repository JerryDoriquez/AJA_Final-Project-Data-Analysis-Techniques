import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration
st.set_page_config(page_title="Titanic Advanced Insights", layout="wide", initial_sidebar_state="expanded")

# Function to load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/JerryDoriquez/AJA_Final-Project-Data-Analysis-Techniques/refs/heads/main/titanic_dataset.csv"
    data = pd.read_csv(url)
    return data

# Load data
data = load_data()

# Sidebar for navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Titanic_survivors.jpg", width=300)
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ['Overview', 'Family Dynamics Analysis', 'Fare Analysis', 'Conclusions'])
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** AJA Team")
st.sidebar.markdown("**Project:** Titanic Advanced Insights")

# Overview Section
if options == 'Overview':
    st.title("Titanic Dataset Advanced Insights")
    st.write("""
    This application dives deeper into the Titanic dataset, focusing on family dynamics, fare analysis, and how these factors influenced passenger groups.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Titanic_-_OP-10527.jpg", caption="RMS Titanic", use_column_width=True)
    st.subheader("Dataset Structure")
    st.write("**Sample Data**")
    st.dataframe(data.head())

    st.subheader("Column Descriptions")
    st.write("""
    - **Survived**: Survival (0 = No, 1 = Yes)
    - **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
    - **Sex**: Gender of the passenger
    - **Age**: Age of the passenger
    - **SibSp**: Number of siblings/spouses aboard
    - **Parch**: Number of parents/children aboard
    - **Fare**: Passenger fare
    """)

# Family Dynamics Analysis Section
elif options == 'Family Dynamics Analysis':
    st.title("Family Dynamics and Survival Analysis")
    st.subheader("Family Size and Survival")

    # Create a family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    st.write("We created a new feature, **FamilySize**, which represents the total number of family members aboard.")
    st.write("Distribution of family size:")
    st.write(data['FamilySize'].value_counts().sort_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data, x='FamilySize', hue='Survived', multiple='stack', kde=False, ax=ax)
    ax.set_title("Survival Based on Family Size")
    st.pyplot(fig)

    st.subheader("Key Insights:")
    st.write("""
    - Passengers with smaller families (1-3 members) had higher survival rates.
    - Larger families faced lower survival rates, likely due to difficulty managing larger groups during evacuation.
    """)

# Fare Analysis Section
elif options == 'Fare Analysis':
    st.title("Fare Distribution and Clustering Analysis")

    st.subheader("Fare Distribution")
    st.write("Examining the distribution of ticket fares among passengers:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data['Fare'], bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Fares")
    st.pyplot(fig)

    st.subheader("Clustering Passengers by Fare")
    st.write("Grouping passengers based on ticket fare and passenger class using K-means clustering.")

    # Feature selection
    fare_features = ['Fare', 'Pclass']
    scaler = StandardScaler()
    scaled_fare_features = scaler.fit_transform(data[fare_features])

    # Determine optimal number of clusters
    inertia = []
    K = range(1, 6)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_fare_features)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)

    st.write("The optimal number of clusters is selected using the Elbow Method. Adjust the slider below to experiment with different cluster counts.")
    optimal_k = st.slider('Select the number of clusters for Fare Analysis', 2, 5, 3)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['FareCluster'] = kmeans.fit_predict(scaled_fare_features)

    st.subheader("Clustered Data Sample")
    st.write(data[['Fare', 'Pclass', 'FareCluster']].head())

    # Visualization of clusters
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_fare_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['FareCluster'] = data['FareCluster']
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='FareCluster', palette='tab10', ax=ax)
    ax.set_title('Fare Clusters Visualization (PCA)')
    st.pyplot(fig)

# Conclusions Section
elif options == 'Conclusions':
    st.title("Conclusions and Recommendations")
    st.write("""
    **Key Takeaways:**
    - Family size significantly influenced survival rates, with smaller families having better outcomes.
    - Fares varied widely across passenger classes, and clustering revealed distinct groups based on economic status.

    **Actionable Recommendations:**
    - Further analyze the relationship between family dynamics and survival strategies.
    - Investigate fare clusters in relation to boarding locations and other socio-economic factors.
    """)

    st.balloons()
    st.sidebar.markdown("---")
