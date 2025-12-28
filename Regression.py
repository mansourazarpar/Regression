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
    # Ensure 'xlrd' is in your requirements.txt
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

fig1, axes = plt.subplots(3, 3, figsize=(14, 10))
# Flatten the 3x3 array of axes to make it easier to loop over
axes = axes.flatten()

for i, name in enumerate(feature_names):
    ax = axes[i]
    ax.scatter(data[name], data["strength"], s=10, alpha=0.6)
    ax.set_xlabel(name)
    ax.set_ylabel("strength")
    ax.set_title(f"{name} vs strength")

# Hide any empty subplots (if features < 9)
for j in range(len(feature_names), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()

# FIX 1: pad_inches=0.2 adds a safety border around the image
st.pyplot(fig1, bbox_inches='tight', pad_inches=0.2)

# -----------------------------------------------
# 4. Correlation bar plot with target variable
# -----------------------------------------------
st.subheader("Correlation Plot")
correlations = data.corr()["strength"].drop("strength")

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.bar(correlations.index, correlations.values, color="green")
plt.xticks(rotation=45, ha="right")
ax2.set_title("Correlation of features with strength")
ax2.set_ylabel("corr(feature, strength)")

# Helper to ensure labels fit before saving
plt.tight_layout()

# FIX 2: pad_inches=0.2 prevents the Y-axis label from being cut
st.pyplot(fig2, bbox_inches='tight', pad_inches=0.2)

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

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(y_test, y_pred, color='blue', alpha=0.7, s=12)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax3.set_xlabel("Actual strength")
ax3.set_ylabel("Predicted strength")
ax3.set_title("Predicted vs Actual (Test Set)")
ax3.grid(True)

plt.tight_layout()

# FIX 3: pad_inches=0.2 prevents "Predicted strength" from being cut
st.pyplot(fig3, bbox_inches='tight', pad_inches=0.2)
