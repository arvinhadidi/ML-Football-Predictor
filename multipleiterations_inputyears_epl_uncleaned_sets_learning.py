import csv
import pandas as pd
import sys
import numpy as np

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def main(nearest, first_year, iterations, rounds):
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python football.py data")

    # Load data from spreadsheet
    data = pd.read_csv(sys.argv[1])

    # Filter the data for testing and training sets based on the specified date range
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
    
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

    # Train model, make predictions and get team points
    model = train_model(evidence_train, labels_train, nearest)
    predictions = model.predict(evidence_test)

    unique_teams, home_teams, away_teams, sorted_team_points = process_data(sys.argv[1], predictions, first_year, second_year)

    keys = [key for key in sorted_team_points]
    values = [sorted_team_points[key] for key in sorted_team_points]
    if iterations == rounds:
        #print("------------------")
        #for i in range(0, len(unique_teams)):
            #print(f"{i+1}: {keys[i]} - {values[i]}") 

        # Print results
        correct = (labels_test == predictions).sum()
        incorrect = (labels_test != predictions).sum()
        total = correct + incorrect
        num_rows = category_sizes(sys.argv[1])
        print("------------------")
        print(f"This algorithm is training itself on data from {round(num_rows - total)} football matches and will be testing itself on {total} matches...")
        print(f"The classification method used is the 'nearest neighbours' method. This classifies using {nearest} nearest neighbours.")
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"The algorithm, with a test proportion of {round(total/num_rows, 2)}, guessed {round((correct/total*100), 2)} % of outcomes of matches the SAME as the ACTUAL SEASON.") 

    return sorted_team_points
     
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


def train_model(evidence, labels, nearest):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    random_seed = 42
    np.random.seed(random_seed)
    neigh = KNeighborsClassifier(n_neighbors=nearest)
    neigh.fit(evidence, labels)

    return neigh


def evaluate(labels, predictions):

    true_positives = 0
    false_positives = 0

    for true_label, predicted_label in zip(labels, predictions):
        if true_label == predicted_label:
            true_positives += 1

        elif true_label != predicted_label:
            false_positives += 1

    return true_positives, false_positives


if __name__ == "__main__":
    nearest_points_list = []  # List to store points for each team across iterations
    rounds = 20
    first_year = int(input("Input the first year of the season you want to test: "))

    nearest_points = {}


    for iterations in range(rounds):
        nearest = iterations+1 * 50  # Update the value of nearest neighbors
        print(f"Iteration {iterations+1}")
        team_points = main(nearest, first_year, iterations, rounds)
        nearest_points_list.append(team_points)  # Store points for this iteration

    # Accumulate team points for all iterations
    for iteration_points in nearest_points_list:
        for team, points in iteration_points.items():
            nearest_points.setdefault(team, 0)
            nearest_points[team] += points

    # Calculate the average points for each team
    average_table = {team: round(points / rounds) for team, points in nearest_points.items()}

    # Sort and print the average table
    sorted_average_table = dict(sorted(average_table.items(), key=lambda x: x[1], reverse=True))
    print("------------------")
    for i, (team, points) in enumerate(sorted_average_table.items(), start=1):
        print(f"{i}: {team} - {points}")
