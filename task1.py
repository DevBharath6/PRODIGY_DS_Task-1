import pandas as pd
import matplotlib.pyplot as plt

# Load the Adult dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, header=None, names=column_names, na_values=' ?')

# Drop rows with missing age values
data = data.dropna(subset=['age'])

# Create a histogram for age distribution
plt.figure(figsize=(10, 6))
plt.hist(data['age'], bins=20, edgecolor='black')
plt.title('Age Distribution in Adult Data Set')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
