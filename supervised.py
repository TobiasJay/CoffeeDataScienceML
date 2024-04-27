import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor




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

estimators = [100, 150, 200, 300, 500, 1000, 1100]
for num_estimators in estimators:
    avg_rmse_train = 0
    avg_rmse_val = 0
    for i in range(30):
        # training: 60%, testing: 20%, validation: 20%
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.4)
        #split the test_val set into test and validation (20% each)
        X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

        # Initialize GBM regressor
        gbm_regressor = GradientBoostingRegressor(n_estimators=num_estimators)

        # Train GBM regressor
        gbm_regressor.fit(X_train, y_train)

        # Make predictions on training and validation sets
        y_train_pred = gbm_regressor.predict(X_train)
        y_val_pred = gbm_regressor.predict(X_val)

        # Calculate RMSE for training set
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Calculate RMSE for validation set
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        #print("Training RMSE:", rmse_train)
        #print("Validation RMSE:", rmse_val)
        avg_rmse_train += rmse_train
        avg_rmse_val += rmse_val
    avg_rmse_train /= 30
    avg_rmse_val /= 30
    print("Number of Estimators:", num_estimators)
    print("Average Training RMSE:", avg_rmse_train)
    print("Average Validation RMSE:", avg_rmse_val)
    print("\n")


'''

# Initialize AdaBoost classifier
ada_clf = AdaBoostClassifier(n_estimators=15, random_state=42)

# Train AdaBoost classifier
ada_clf.fit(X_train, y_train)

# Make predictions on the testing/validation set
ada_preds = ada_clf.predict(X_val)

# Calculate accuracy of AdaBoost classifier
ada_mse = mean_squared_error(y_val, ada_preds)
ada_mse_test = mean_squared_error(y_test, ada_clf.predict(X_test))

print("Validation AdaBoost Root Mean Squared Error:", np.sqrt(ada_mse))
print("Test AdaBoost Root Mean Squared Error:", np.sqrt(ada_mse_test))

# Initialize Random Forest classifier
rf_clf = RandomForestClassifier()

# Train Random Forest classifier
rf_clf.fit(X_train, y_train)

# Make predictions on the testing/validation set
rf_preds = rf_clf.predict(X_val)

# Calculate accuracy of Random Forest classifier
rf_mse = mean_squared_error(y_val, rf_preds)
rf_mse_test = mean_squared_error(y_test, rf_clf.predict(X_test))
print("Validation Random Forest Root Mean Squared Error:", np.sqrt(rf_mse))    
print("Test Random Forest Root Mean Squared Error:", np.sqrt(rf_mse_test))
'''












