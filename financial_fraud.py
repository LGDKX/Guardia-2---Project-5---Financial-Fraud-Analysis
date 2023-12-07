#########################################################################################
#             Importing pandas as pd in order to manipulate and explore data            #
#                     Import random in order to generate random data                    #
#                   Import string in order to manipulate data strings                   #
#                  Import os in order to save files in another folder                   #
#         Import train_test_split from Scikit Learn in order to break data sets         #
#        Import MAE from Scikit Learn in order to assure the quality of the model       #
#             Import LabelEncoder from Scikit Learn in order to encode data             #
#                            Import Machin Learning algorithm                           #
#########################################################################################
import pandas as pd
import random
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.gaussian_process import GaussianProcessClassifier



def data_set_choices():
    print("Which Data Set do you wish to use ?")
    print("1 - financial_fraud (The complete Data Set.)")
    print("2 - financial_fraud_reduced (The Data Set that have been reduced.)")
    print("3 - financial_fraud_incomplete (The Data Set containing error which have been cleaned.)")
    print("4 - random_data (A previously randomly generated Data Set. This Data Set should only serve to predict.))")
    print("5 - Custom (Allows you to choose a custom Data Set. This Data Set should only serve to predict.)")
    choice = input("Your choice : ")


def function_choices():
    print("What do you want to do ?")
    print("1 - Train, predict and evaluate")
    print("2 - Predict and evaluate only")
    choice = input("Your choice : ")


def files_configuration(data_set, training):
    # Save filepath for easier access
    file_path = 'data_set/' + data_set

    # Read the data and save it into a DataFrame titled file_data
    file_data = pd.read_csv(file_path)

    # Init the LabelEncoder() function in order to encode data
    label_encoder = LabelEncoder()

    # Convert categorical data into numerical data usable with the model
    file_data['type'] = label_encoder.fit_transform(file_data['type'])
    file_data['nameOrig'] = label_encoder.fit_transform(file_data['nameOrig'])
    file_data['nameDest'] = label_encoder.fit_transform(file_data['nameDest'])

    # Select the control data
    y = file_data['isFraud']

    # Select the whole table apart from de the isFraud column
    X = file_data.drop('isFraud', axis=1)

    # Separate the data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Print the available choices
    print("Which model do you want to use ?")
    print("1 - Linear Regression")
    print("2 - Logistic Regression")
    print("3 - Ridge Regression")
    print("4 - Lasso Regression")
    print("5 - Decision Tree")
    print("6 - Random Forest")
    print("7 - Gradient Boosting Regression")
    print("8 - Extreme Gradient Boosting")
    print("9 - LightGBM Regressor")
    print("10 - KMeans")
    print("11 - Hierarchical Clustering")
    print("12 - Gaussian Process")
    print("13 - APriori Algorithm")
    print("14 - All of them")
    print("15 - Do something else")

    while True:
        # Input your choice
        choice = input("Your choice : ")
        if choice == "1":
            linear_regression_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "2":
            logistic_regression_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "3":
            ridge_regression_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "4":
            lasso_regression_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "5":
            decision_tree_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "6":
            random_forest_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "7":
            gradient_boosting_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "8":
            xgboost_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "9":
            lightgbm_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "10":
            kmeans_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "11":
            hierarchical_clustering_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "12":
            gaussian_process_model(training, train_X, train_y, val_X, val_y, X, y)
        elif choice == "13":
            print("Fuck")
        elif choice == "14":
            models = [
                linear_regression_model,
                logistic_regression_model,
                ridge_regression_model,
                lasso_regression_model,
                decision_tree_model,
                random_forest_model,
                gradient_boosting_model,
                xgboost_model,
                lightgbm_model,
                kmeans_model,
                hierarchical_clustering_model,
                gaussian_process_model,
            ]
            for model_func in models:
                model_func(training, train_X, train_y, val_X, val_y, X, y)
        else:
            break


def linear_regression_model(training, train_X, train_y, val_X, val_y, X, y):
    model = LinearRegression()
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def logistic_regression_model(training, train_X, train_y, val_X, val_y, X, y):
    model = LogisticRegression()
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def ridge_regression_model(training, train_X, train_y, val_X, val_y, X, y):
    model = Ridge()
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def lasso_regression_model(training, train_X, train_y, val_X, val_y, X, y):
    model = Lasso()
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def decision_tree_model(training, train_X, train_y, val_X, val_y, X, y):
    model = DecisionTreeClassifier(random_state=1)
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def random_forest_model(training, train_X, train_y, val_X, val_y, X, y):
    model = RandomForestClassifier(random_state=1)
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def gradient_boosting_model(training, train_X, train_y, val_X, val_y, X, y):
    model = GradientBoostingClassifier(random_state=1)
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def xgboost_model(training, train_X, train_y, val_X, val_y, X, y):
    model = XGBRegressor(random_state=1)
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def lightgbm_model(training, train_X, train_y, val_X, val_y, X, y):
    model = LGBMRegressor(random_state=1)
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def kmeans_model(training, train_X, train_y, val_X, val_y, X, y):
    model = KMeans(n_clusters=2, random_state=1, n_init="auto")
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def hierarchical_clustering_model(training, train_X, train_y, val_X, val_y, X, y):
    model = AgglomerativeClustering()
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)


def gaussian_process_model(training, train_X, train_y, val_X, val_y, X, y):
    model = GaussianProcessClassifier(random_state=1)
    fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y)

# Helper function to fit, predict, and print Mean Absolute Error


def fit_predict_print(model, training, train_X, train_y, val_X, val_y, X, y):
    if training:
        model.fit(train_X, train_y)
        prediction = model.predict(val_X)
        mae = mean_absolute_error(val_y, prediction)
        print(mae)
    else:
        prediction = model.predict(X)
        mae = mean_absolute_error(y, prediction)
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
        print("1 - Train Machine Learning algorithms, predict and check precision")
        print("2 - Create a new random Data Set")
        print("3 - End script")
        choice = input("Your choice : ")
        if choice == "1":
            data_set_choices()
            if choice == "1":
                data_set = "financial_fraud.csv"
                function_choices()
                if choice == "1":
                    training = True
                    files_configuration(data_set, training)
                else:
                    training = False
                    files_configuration(data_set, training)
            elif choice == "2":
                data_set = "financial_fraud_reduced.csv"
                function_choices()
                if choice == "1":
                    training = True
                    files_configuration(data_set, training)
                else:
                    training = False
                    files_configuration(data_set, training)
            elif choice == "3":
                data_set = "financial_fraud_incomplete.csv"
                function_choices()
                if choice == "1":
                    training = True
                    files_configuration(data_set, training)
                else:
                    training = False
                    files_configuration(data_set, training)
            elif choice == "4":
                data_set = "random_data.csv"
                training = False
                files_configuration(data_set, training)
            elif choice == "5":
                data_set = input("Please enter the name of your Data Set : ")
                training = False
                files_configuration(data_set, training)
        elif choice == "2":
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
