import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('../data/student_data.csv')

# Features and target
X = data[['hours_studied', 'sleep_hours', 'attendance', 'previous_score']]
y = data['final_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
sample = [[5, 7, 85, 70]]
prediction = model.predict(sample)

print("Predicted Score:", prediction[0])
