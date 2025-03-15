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

    # Extract features and labels
    X = df_labelled.drop(columns=["city", "year"]).values  # Remove non-feature columns
    y = df_labelled["city"].values  # Target labels
    X_unlabelled = df_unlabelled.drop(columns=["year", "city"]).values  # No city column

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_unlabelled_scaled = scaler.transform(X_unlabelled)

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    best_model = None
    best_accuracy = 0
    best_predictions = None

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_valid = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, y_pred_valid)
        print(f"{name} Accuracy: {accuracy:.4f}")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_predictions = model.predict(X_unlabelled_scaled)

    # Save predictions from the best model
    pd.Series(best_predictions).to_csv(output_file, index=False, header=False)
    print(f"Best model: {best_model.__class__.__name__} with accuracy {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
