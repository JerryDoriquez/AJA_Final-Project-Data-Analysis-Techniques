import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report




@st.cache_data
def load_data():
    
    url = "https://raw.githubusercontent.com/JerryDoriquez/AJA_Final-Project-Data-Analysis-Techniques/refs/heads/main/titanic_dataset.csv"  
    data = pd.read_csv(url)
    return data


data = load_data()
st.title("Titanic Dataset Analysis")
st.subheader("Overview of the Dataset")
st.write(data.head())


st.title("Data Exploration")
st.write("### Basic Statistics")
st.write(data.describe())

st.write("### Missing Values")
missing_values = data.isnull().sum()
st.write(missing_values)


st.write("### Heatmap of Missing Values")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
st.pyplot(fig)



st.title("Data Preparation")


data['Age'] = data['Age'].fillna(data['Age'].median())


data = data.dropna(subset=['Embarked'])


data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


features = ['Pclass', 'Age', 'Fare', 'Sex']
X = data[features]


st.write("### Prepared Data")
st.write(X.head())


st.title("K-means Clustering")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


st.subheader("K-means Clustering")
st.write("### Cluster Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Fare'], hue=data['Cluster'], palette='viridis')
plt.title("Clusters Based on Age and Fare")
st.pyplot(fig)



st.title("Linear Regression")


y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
st.write(f"### Model Accuracy: {accuracy:.2f}")


y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)


st.write("### Classification Report")
st.write(pd.DataFrame(report).transpose())


coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, orient='h')
plt.title("Feature Importance")
st.pyplot(fig)



st.title("Interactive Visualization")


min_age, max_age = st.slider("Select Age Range", int(data['Age'].min()), int(data['Age'].max()), (20, 50))
filtered_data = data[(data['Age'] >= min_age) & (data['Age'] <= max_age)]


st.write(f"### Passengers Aged Between {min_age} and {max_age}")
st.write(filtered_data[['Age', 'Fare', 'Pclass', 'Survived']].head())


fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=filtered_data['Age'], y=filtered_data['Fare'], hue=filtered_data['Survived'], palette='coolwarm')
plt.title(f"Passengers Aged Between {min_age} and {max_age}")
st.pyplot(fig)



st.title("Conclusions and Recommendations")
st.write("""
- **Cluster Insights**: Passengers were grouped into clusters based on age, fare, and class. The analysis showed clear groupings based on socio-economic status.
- **Survival Factors**: Logistic regression revealed that women and passengers in higher classes had significantly better chances of survival.
- **Actionable Insights**:
  - Future evacuation protocols could prioritize high-risk groups such as older passengers and those in lower classes.
  - Highlight the importance of socio-economic disparities in survival rates.
""")



