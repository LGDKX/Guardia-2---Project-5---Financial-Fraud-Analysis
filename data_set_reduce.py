import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Charger les données
data = pd.read_csv("data_set/financial_fraud.csv")

# Init the LabelEncoder() function in order to encode data
label_encoder = LabelEncoder()

# Convert categorical data into numerical data usable with the model
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Séparer les caractéristiques (features) et la variable cible
X = data.drop(['isFraud'], axis=1)  # Features
y = data['isFraud']  # Target variable

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer PCA
n_components = 3
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Créer un nouveau DataFrame avec les données réduites en dimension
columns = [f'PC{i}' for i in range(1, n_components + 1)]
X_pca_df = pd.DataFrame(data=X_pca, columns=columns)

# Ajouter la variable cible au DataFrame réduit en dimension
X_pca_df['isFraud'] = y

# Enregistrer dans un fichier CSV
X_pca_df.to_csv("data_set/financial_fraud_reduced.csv", index=False)
