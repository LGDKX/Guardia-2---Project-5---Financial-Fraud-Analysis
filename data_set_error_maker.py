import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Charger le fichier CSV
df = pd.read_csv('data_set/financial_fraud.csv')

# Init the LabelEncoder() function in order to encode data
label_encoder = LabelEncoder()

# Convert categorical data into numerical data usable with the model
df['type'] = label_encoder.fit_transform(df['type'])
df['nameOrig'] = label_encoder.fit_transform(df['nameOrig'])
df['nameDest'] = label_encoder.fit_transform(df['nameDest'])

# Définir la proportion de données manquantes que vous souhaitez
missing_quantity = 0.1  # 10% de données manquantes

# Create a random mask for missing data
mask = np.random.rand(*df.shape) < missing_quantity

# Input blank data where mask is True
df[mask] = ''

# Save new csv file with missing data
df.to_csv('data_set/financial_fraud_incomplete.csv', index=False)