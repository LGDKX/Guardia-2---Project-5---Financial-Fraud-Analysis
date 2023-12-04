##################################################################
# Importing pandas as pd in order to manipulate and explore data #
##################################################################
import pandas as pd

#########################################################################
# Import train_test_split from Scikit Learn in order to break data sets #
#########################################################################
from sklearn.model_selection import train_test_split

############################################################################
# Import MAE from Scikit Learn in order to assure the quality of the model #
############################################################################
from sklearn.metrics import mean_absolute_error

#################################################################
# Import LabelEncoder from Scikit Learn in order to encode data #
#################################################################
from sklearn.preprocessing import LabelEncoder

########################################################################################
# Import DecisionTreeClassifier from Scikit Learn in order to manipulate Decision Tree #
########################################################################################
from sklearn.tree import DecisionTreeClassifier

# Save filepath for easier access
financial_fraud_file_path = 'data_set/financial_fraud.csv'

# Read the data and save it into a DataFrame titled financial_fraud_data
financial_fraud_data = pd.read_csv(financial_fraud_file_path)

# Init the LabelEncoder() function in order to encode data
label_encoder = LabelEncoder()

# Convert categorical data into numerical data usable with the model
financial_fraud_data['type'] = label_encoder.fit_transform(financial_fraud_data['type'])
financial_fraud_data['nameOrig'] = label_encoder.fit_transform(financial_fraud_data['nameOrig'])
financial_fraud_data['nameDest'] = label_encoder.fit_transform(financial_fraud_data['nameDest'])

# Select the control data
y = financial_fraud_data.isFraud

# Select the whole table apart from de the isFraud column
X = financial_fraud_data.drop('isFraud', axis=1)

# Separate the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define model. Specify a number for random_state to ensure same results each run
financial_fraud_model = DecisionTreeClassifier(random_state=1)

# Fit model
financial_fraud_model.fit(train_X, train_y)

# Prediction launch message
print("Making predictions for ...")
# Print the features
print(val_X.head())

# Define the prediction
prediction = financial_fraud_model.predict(X)

# Print the predictions
print("The prediction are : ")
print(prediction)

# Define the Mean Absolute Error
mae = mean_absolute_error(val_y, prediction)
# Print the Mean Absolute Error
print(mae)
