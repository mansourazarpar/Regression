# Assignment 1 web streamlit
# – Regression with MSE and Feature Correlation
# Author: Student-style code – clear and commented

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st

# -----------------------------------------------
# 0. App Title
# -----------------------------------------------
st.title("My Regression Assignment 📊")

# -----------------------------------------------
# 1. Load dataset from UCI
# -----------------------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url)
    df.columns = [
        "cement", "slag", "fly_ash", "water", "superplasticizer",
        "coarse_agg", "fine_agg", "age", "strength"
    ]
    return df

try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Show the raw data (Optional)
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# -----------------------------------------------
# 2. Prepare input and output
# -----------------------------------------------
X = data.drop(columns="strength").values
y = data["strength"].values
feature_names = data.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------------------------
# 3. Scatter plots for feature vs strength
# -----------------------------------------------
st.subheader("Feature Scatter Plots")

# FIX: Added layout='constrained'. This guarantees labels fit inside the box.
fig1 = plt.figure(figsize=(14, 10), layout='constrained')

for i, name in enumerate(feature_names, start=1):
    plt.subplot(3, 3, i)
    plt.scatter(data[name], data["strength"], s=10, alpha=0.6)
    plt.xlabel(name)
    plt.ylabel("strength")
    plt.title(f"{name} vs strength")

st.pyplot(fig1)

# -----------------------------------------------
# 4. Correlation bar plot with target variable
# -----------------------------------------------
st.subheader("Correlation Plot")
correlations = data.corr()["strength"].drop("strength")

# FIX: Added layout='constrained'
fig2 = plt.figure(figsize=(8, 6), layout='constrained')

plt.bar(correlations.index, correlations.values, color="green")
plt.xticks(rotation=45, ha="right")
plt.title("Correlation of features with strength")
plt.ylabel("corr(feature, strength)")

st.pyplot(fig2)

# -----------------------------------------------
# 5. Fit linear regression
# -----------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------
# 6. Evaluate with Mean Squared Error
# -----------------------------------------------
mse = mean_squared_error(y_test, y_pred)
st.success(f"Mean Squared Error (Test Set): {mse:.3f}")

# -----------------------------------------------
# 7. Plot predicted vs actual
# -----------------------------------------------
st.subheader("Predictions vs Actual")

# FIX: Added layout='constrained'
fig3 = plt.figure(figsize=(8, 6), layout='constrained')

plt.scatter(y_test, y_pred, color='blue', alpha=0.7, s=12)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual strength")
plt.ylabel("Predicted strength")
plt.title("Predicted vs Actual (Test Set)")
plt.grid(True)

st.pyplot(fig3)
