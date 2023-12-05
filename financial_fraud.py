#########################################################################################
#             Importing pandas as pd in order to manipulate and explore data            #
#                     Import random in order to generate random data                    #
#                   Import string in order to manipulate data strings                   #
#                  Import os in order to save files in another folder                   #
#         Import train_test_split from Scikit Learn in order to break data sets         #
#        Import MAE from Scikit Learn in order to assure the quality of the model       #
#             Import LabelEncoder from Scikit Learn in order to encode data             #
# Import DecisionTreeClassifier from Scikit Learn in order to manipulate Decision Tree  #
#########################################################################################
import pandas as pd
import random
import string
import os
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


def generate_random_data(entry):
    steps = list(range(1, entry + 1))
    types = ["PAYMENT", "TRANSFER", "CASH_OUT"]
    amounts = [round(random.uniform(1, 10000), 2) for _ in range(entry)]
    names = [''.join(random.choices(string.ascii_uppercase, k=10)) for _ in range(entry)]
    orig_balances = [round(random.uniform(1, 100000), 2) for _ in range(entry)]
    dest_names = [''.join(random.choices(string.ascii_uppercase, k=10)) for _ in range(entry)]
    dest_balances = [round(random.uniform(1, 100000), 2) for _ in range(entry)]
    is_fraud = [random.choice([0, 1]) for _ in range(entry)]
    is_flagged_fraud = [random.choice([0, 1]) for _ in range(entry)]

    data = {
        'step': steps,
        'type': [random.choice(types) for _ in range(entry)],
        'amount': amounts,
        'nameOrig': names,
        'oldbalanceOrg': orig_balances,
        'newbalanceOrig': [ob - amt for ob, amt in zip(orig_balances, amounts)],
        'nameDest': dest_names,
        'oldbalanceDest': dest_balances,
        'newbalanceDest': [nb + amt for nb, amt in zip(dest_balances, amounts)],
        'isFraud': is_fraud,
        'isFlaggedFraud': is_flagged_fraud
    }

    df = pd.DataFrame(data)
    return df


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
                df = generate_random_data(entry)

                save_directory = './data_set'

                # Save the DataFrame to a CSV file in the specified directory
                file_path = os.path.join(save_directory, 'random_data.csv')
                df.to_csv(file_path, index=False)
                print(f"CSV file created successfully at {file_path}.")
        else:
            # Quit the function
            break


# Calling main function
main()


