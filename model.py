import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Input feature
X = data[['Hours']]

# Output label
y = data['Marks']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully")
