#########################################################################################
#             Importing pandas as pd in order to manipulate and explore data            #
#         Import train_test_split from Scikit Learn in order to break data sets         #
#        Import MAE from Scikit Learn in order to assure the quality of the model       #
#             Import LabelEncoder from Scikit Learn in order to encode data             #
# Import DecisionTreeClassifier from Scikit Learn in order to manipulate Decision Tree  #
#########################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

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
y = financial_fraud_data['isFraud']

# Select the whole table apart from de the isFraud column
X = financial_fraud_data.drop('isFraud', axis=1)

# Separate the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


def model_choices():
    # Print the available choices
    print("Which model do you want to use ?")
    print("1 - Decision Tree Classifier")
    print("2 -")
    print("3 -")
    print("4 -")
    print("5 -")
    print("6 -")
    print("7 -")
    print("8 -")
    print("9 -")
    print("10 - All of them")
    # Input your choice
    choice = input("Your choice : ")


def decision_tree_classifier_model():
    # Define model. Specify a number for random_state to ensure same results each run
    financial_fraud_model = tree.DecisionTreeClassifier(random_state=1)

    # Fit model
    financial_fraud_model.fit(train_X, train_y)

    # Define the prediction
    prediction = financial_fraud_model.predict(val_X)

    # Define the Mean Absolute Error
    mae = mean_absolute_error(val_y, prediction)
    # Print the Mean Absolute Error
    print(mae)


def main():
    while True:
        # Main menu
        print("Please choose something to do : ")
        print("1 - Train Machine Learning algorithms and check precision")
        print("2 - Predict with another Data Set (the Data Set should be inside the folder data_set and be a .csv")
        print("3 - Create a new random Data Set")
        print("4 - End script")
        choice = input("Your choice : ")
        if choice == "1":
            # Choosing the model to train
            model_choices()
            if choice == "1":
                decision_tree_classifier_model()
        elif choice == "2":
            # Choosing the Data Set to use
            print("Please input the name of the Data Set (without file extension")
            new_data_set = input("Your Data Set name : ") + ".csv"
        elif choice == "3":
            # Choosing the number of entry for the new Data Set
            print("How much entry do you want ?")
            entry = int(input("Your choice : "))
            if entry < 0:
                # Error message
                print("Invalid number")
            else:
                print("Function in construction")
        else:
            # Quit the function
            break


# Calling main function
main()


