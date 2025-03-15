import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load labelled dataset
    labelled_file = sys.argv[1]
    unlabelled_file = sys.argv[2]
    output_file = sys.argv[3]

    df_labelled = pd.read_csv(labelled_file)
    df_unlabelled = pd.read_csv(unlabelled_file)

    print(df_unlabelled)
    # Extract features and labels
    X = df_labelled.drop(columns=["city", "year"]).values  # Remove non-feature columns
    y = df_labelled["city"].values  # Target labels

    X_unlabelled = df_unlabelled.drop(columns=["year", "city"]).values  # No city column
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_unlabelled_scaled = scaler.transform(X_unlabelled)

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train classifier (Random Forest)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    # model = GaussianNB()
    # model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred_valid = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred_valid)
    print(f"{accuracy:.4f}")  # Print validation accuracy

    # Predict cities for the unlabelled data
    predictions = model.predict(X_unlabelled_scaled)

    # Save predictions
    pd.Series(predictions).to_csv(output_file, index=False, header=False)

if __name__ == "__main__":
    main()
