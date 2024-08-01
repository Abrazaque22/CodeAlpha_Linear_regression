import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
@st.cache_data
def load_data():
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = boston.target
    return X, y, boston.feature_names

X, y, feature_names = load_data()

st.title("Boston Housing Price Prediction")

st.write("This app uses linear regression to predict housing prices in Boston based on various features.")

# Sidebar for user input
st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.header("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
col3.metric("R-squared Score", f"{r2:.2f}")

# Feature coefficients
st.header("Feature Coefficients")
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
st.dataframe(coef_df)

# Visualizations
st.header("Visualizations")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted Housing Prices")
st.pyplot(fig)

# Residual Plot
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred, residuals, alpha=0.5)
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")
ax.axhline(y=0, color='r', linestyle='--')
st.pyplot(fig)

# Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': np.abs(model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)
ax.bar(feature_importance['feature'], feature_importance['importance'])
ax.set_xlabel("Features")
ax.set_ylabel("Absolute Coefficient Value")
ax.set_title("Feature Importance")
plt.xticks(rotation=90)
st.pyplot(fig)

# Price Prediction
st.header("Price Prediction")
st.write("Use the sliders below to input feature values and get a price prediction.")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

user_input_df = pd.DataFrame([user_input])
prediction = model.predict(user_input_df)[0]

st.subheader(f"Predicted Price: ${prediction:.2f}k")