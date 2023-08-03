import csv
import pandas as pd
import sys

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python football.py data")

    # Load data from spreadsheet
    data = pd.read_csv(sys.argv[1])

    # Filter the data for testing and training sets based on the specified date range
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")

    first_year = int(input("Input the first year of the season you want to test: "))
    second_year = first_year + 1
    start_date = "01/08/" + str(first_year)
    end_date = "01/08/" + str(second_year)

    start_datetime = datetime.strptime(start_date, "%d/%m/%Y")
    end_datetime = datetime.strptime(end_date, "%d/%m/%Y")

    test_data = data[(data['Date'] >= start_datetime) & (data['Date'] <= end_datetime)]
    train_data = data[(data['Date'] < start_datetime) | (data['Date'] > end_datetime)]

    # Load data into evidence and labels for training and testing sets
    evidence_train, labels_train = load_data(train_data)
    evidence_test, labels_test = load_data(test_data)

    # Train model and make predictions
    model = train_model(evidence_train, labels_train)
    predictions = model.predict(evidence_test)

    # Get team points based on model predictions
    unique_teams, home_teams, away_teams, sorted_team_points = process_data(sys.argv[1], predictions, first_year, second_year)

    # Print the team points and results
    keys = [key for key in sorted_team_points]
    values = [sorted_team_points[key] for key in sorted_team_points]
    print("------------------")
    for i in range(0, len(unique_teams)):
        print(f"{i+1}: {keys[i]} - {values[i]}") 

    correct = (labels_test == predictions).sum()
    incorrect = (labels_test != predictions).sum()
    total = correct + incorrect
    num_rows = category_sizes(sys.argv[1])
    print("------------------")
    print(f"This algorithm is training itself on data from {round(num_rows - total)} football matches and will be testing itself on {total} matches...")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"The algorithm, with a test proportion of {round(total/num_rows, 2)}, guessed {round((correct/total*100), 2)} % of outcomes of matches the SAME as the ACTUAL SEASON.") 
 
def process_data(filename, predictions, first_year, second_year):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Filter the DataFrame to include only the dates between 01/08/201n and 01/08/201n+1
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    start_date = str(first_year) + '-08-01'
    end_date = str(second_year) + '-08-01'
    df = df[(df['Date'] >= start_date) & (df['Date'] < end_date)]

    # Create empty lists to store unique teams and the three lists
    unique_teams = []
    home_teams = []
    away_teams = []

    # Create an empty dictionary to store team points
    team_points = {}
    
    # Iterate through the DataFrame and extract the required data
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        if home_team not in unique_teams:
            unique_teams.append(home_team)
            # Initialize points to zero for each team
            team_points[home_team] = 0

        # Append the data to the lists
        home_teams.append(home_team)
        away_teams.append(away_team)

    # Update team points based on the predictions
    for i, prediction in enumerate(predictions):
        home_team = home_teams[i]
        away_team = away_teams[i]

        if prediction == 0:
            team_points[home_team] += 3
        elif prediction == 1:
            team_points[away_team] += 3
        else:
            team_points[home_team] += 1
            team_points[away_team] += 1

    sorted_team_points = dict(sorted(team_points.items(), key=lambda x: x[1], reverse=True))

    return unique_teams, home_teams, away_teams, sorted_team_points

def category_sizes(filename):
    df = pd.read_csv(filename)
    num_rows = df.shape[0]
    return num_rows


def load_data(data):
    """
    Load football data from a DataFrame `data` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    FTR = {'H': 0, 'A': 1, 'D': 2}

    labels_df = data['FTR']  # Extract the 'FTR' column

    evidence_df = data.drop(columns=['FTR'])
    numerical_columns = evidence_df.select_dtypes(include=[float, int]).columns

    evidence_df = evidence_df[numerical_columns]

    evidence_list = evidence_df.values.tolist()
    labels_list = labels_df.replace(FTR).values.tolist()

    return evidence_list, labels_list


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model trained on the data.
    """
    nearest = int(input("number of nearest neighbours? "))
    neigh = KNeighborsClassifier(n_neighbors=nearest)
    neigh.fit(evidence, labels)

    return neigh


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


if __name__ == "__main__":
    main()
