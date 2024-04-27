import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


original_df = pd.read_csv('data/PairIDdata.csv')

# Assuming 'original_df' is your original DataFrame with the 'transaction_time' column

# Convert 'transaction_time' to datetime format if it's not already in datetime format
original_df['transaction_date'] = pd.to_datetime(original_df['transaction_date'])

# Extract year, month, day, and day of the week from the 'transaction_time' column
original_df['date'] = original_df['transaction_date']
original_df['year'] = original_df['transaction_date'].dt.year
original_df['month'] = original_df['transaction_date'].dt.month
original_df['day_of_month'] = original_df['transaction_date'].dt.day
original_df['day_of_week'] = original_df['transaction_date'].dt.dayofweek

# Group by year, month, day of the month, and day of the week, and count the number of rows
grouped_df = original_df.groupby(['date', 'year', 'month', 'day_of_month', 'day_of_week']).size().reset_index(name='orders_per_day')

# Plot the number of rows per day
'''
plt.figure(figsize=(12, 6))
plt.plot(grouped_df['date'], grouped_df['orders_per_day'])
plt.xlabel('Date')
plt.ylabel('Orders per Day')
plt.title('Orders per Day over time')
plt.show()
'''

# Use only 3 features. Month, day of the month, and day of the week. Its all the same year so we don't need that. and we don't need the date since its encapsulated in the other 3 features
ml_ready_df = grouped_df[['month', 'day_of_month', 'day_of_week', 'orders_per_day']]
X = ml_ready_df.drop('orders_per_day', axis=1)
y = ml_ready_df['orders_per_day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize AdaBoost classifier
ada_clf = AdaBoostClassifier()

# Train AdaBoost classifier
ada_clf.fit(X_train, y_train)

# Make predictions on the testing/validation set
ada_preds = ada_clf.predict(X_test)

# Calculate accuracy of AdaBoost classifier
ada_accuracy = accuracy_score(y_test, ada_preds)
print("AdaBoost Classifier Accuracy:", ada_accuracy)

# Initialize Random Forest classifier
rf_clf = RandomForestClassifier()

# Train Random Forest classifier
rf_clf.fit(X_train, y_train)

# Make predictions on the testing/validation set
rf_preds = rf_clf.predict(X_test)

# Calculate accuracy of Random Forest classifier
print(rf_preds - y_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print("Random Forest Classifier Accuracy:", rf_accuracy)