import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/PairIDdata.csv')

# Graph the number of times each product was purchased with time on the x-axis
plt.figure(figsize=(10, 6))
data['product_id'].value_counts().plot(kind='bar')
print(data['product_id'].value_counts())


plt.show()