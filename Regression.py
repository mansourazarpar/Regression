# Assignment 1 web streamlit
# – Regression with MSE and Feature Correlation
# Author: Student-style code – clear and commented

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st  # <--- CHANGED 1: Added this library

# -----------------------------------------------
# 0. App Title (Optional but good for web)
# -----------------------------------------------
st.title("My Regression Assignment 📊") # <--- Added this so the page has a name

# -----------------------------------------------
# 1. Load dataset from UCI
# -----------------------------------------------
@st.cache_data  # <--- OPTIONAL: This makes the app load faster
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url)
    df.columns = [
        "cement", "slag", "fly_ash", "water", "superplasticizer",
        "coarse_agg", "fine_agg", "age", "strength"
    ]
    return df

data = load_data()

# Show the raw data to your friend (Optional)
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
st.subheader("Feature Scatter Plots") # <--- Added a header
plt.figure(figsize=(14, 10))
for i, name in enumerate(feature_names, start=1):
    plt.subplot(3, 3, i)
    plt.scatter(data[name], data["strength"], s=10, alpha=0.6)
    plt.xlabel(name)
    plt.ylabel("strength")
    plt.title(f"{name} vs strength")
plt.tight_layout()
st.pyplot(plt) # <--- CHANGED 2: Replaced plt.show()

# -----------------------------------------------
# 4. Correlation bar plot with target variable
# -----------------------------------------------
st.subheader("Correlation Plot") # <--- Added a header
correlations = data.corr()["strength"].drop("strength")
plt.figure(figsize=(7, 4))
plt.bar(correlations.index, correlations.values, color="green")
plt.xticks(rotation=45, ha="right")
plt.title("Correlation of features with strength")
plt.ylabel("corr(feature, strength)")
plt.tight_layout()
st.pyplot(plt) # <--- CHANGED 2: Replaced plt.show()

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
st.success(f"Mean Squared Error (Test Set): {mse:.3f}") # <--- CHANGED 3: Replaced print()

# -----------------------------------------------
# 7. Plot predicted vs actual
# -----------------------------------------------
st.subheader("Predictions vs Actual") # <--- Added a header
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, s=12)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual strength")
plt.ylabel("Predicted strength")
plt.title("Predicted vs Actual (Test Set)")
plt.grid(True)
plt.tight_layout()
st.pyplot(plt) # <--- CHANGED 2: Replaced plt.show()