import csv
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = float(input("proportion of dataset to be tested? "))


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, shuffle = True
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    true_positives, false_positives = evaluate(y_test, predictions)

    # Print results
    correct = (y_test == predictions).sum()
    incorrect = (y_test != predictions).sum()
    total = correct + incorrect
    num_rows = category_sizes(sys.argv[1])
    print(f"This algorithm is training itself on data from {round(num_rows - total)} football matches and will be testing itself on {total} matches...")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"The algorithm, with a test proportion of {TEST_SIZE}, guessed {round((correct/total*100), 2)} % of outcomes of matches correctly.") 

def category_sizes(filename):
    df = pd.read_csv(filename)
    num_rows = df.shape[0]

    return num_rows

def load_data(filename):

    evidence = []
    labels = []

    FTR = {'H': 0, 'A': 1, 'D': 2}

    csv_file = pd.read_csv(filename)

    # Clean the data set (remove all string columns apart from FTR)
    non_numeric_columns = csv_file.select_dtypes(exclude=[float, int]).drop(columns=['FTR']).columns
    csv_file = csv_file.drop(non_numeric_columns, axis=1)

    # Extract the 'FTR' column
    labels_df = csv_file['FTR']  
    evidence_df = csv_file.drop(columns=['FTR'])
    
    evidence_df = evidence_df.replace(FTR)
    labels_df = labels_df.replace(FTR)
    
    evidence_list = evidence_df.values.tolist()
    labels_list = labels_df.values.tolist()

    return evidence_list, labels_list


    raise NotImplementedError


def train_model(evidence, labels):
    
    nearest = int(input("number of nearest neighbours? "))
    neigh = KNeighborsClassifier(n_neighbors = nearest)
    neigh.fit(evidence, labels)
    
    return neigh

    raise NotImplementedError


def evaluate(labels, predictions):

    # Evaluate how many predictions were the same as the real result
    true_positives = 0
    false_positives = 0

    for true_label, predicted_label in zip(labels, predictions):
        if true_label == 1 and predicted_label == 1:
            true_positives += 1
        elif true_label == 2 and predicted_label == 2:
            true_positives += 1
        elif true_label == 3 and predicted_label == 3:
            true_positives += 1
            
        if true_label == 1 and predicted_label != 1:
            false_positives += 1
        elif true_label == 2 and predicted_label != 2:
            false_positives += 1
        elif true_label == 3 and predicted_label != 3:
            false_positives += 1

    return true_positives, false_positives
    
    raise NotImplementedError


if __name__ == "__main__":
    main()
