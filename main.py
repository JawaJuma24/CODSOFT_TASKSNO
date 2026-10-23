"""
CodSoft Data Science Internship - Task 1
Titanic Survival Prediction

Dataset expected: Kaggle Titanic train.csv
Run: python main.py
"""

from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("train.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Could not find train.csv. Download the Kaggle Titanic training "
            "dataset, place train.csv in this project folder, and run again."
        )

    data = pd.read_csv(path)
    if "Survived" not in data.columns:
        raise ValueError(
            "The CSV must contain the 'Survived' target column. "
            "Use the Kaggle Titanic train.csv file, not test.csv."
        )
    return data


def main():
    data = load_data(DATA_FILE)

    # Select useful passenger features and the target.
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = data[features]
    y = data["Survived"]

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== TITANIC SURVIVAL PREDICTION ===")
    print(f"Total passengers used: {len(data)}")
    print(f"Training passengers: {len(X_train)}")
    print(f"Testing passengers: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Did not survive", "Survived"]))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Demonstration prediction for a hypothetical passenger.
    sample_passenger = pd.DataFrame(
        [{
            "Pclass": 3,
            "Sex": "male",
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S",
        }]
    )

    sample_prediction = model.predict(sample_passenger)[0]
    sample_probability = model.predict_proba(sample_passenger)[0][1]

    print("\nExample passenger:")
    print(sample_passenger.to_string(index=False))
    print("Predicted result:", "Survived" if sample_prediction == 1 else "Did not survive")
    print(f"Model-estimated survival probability: {sample_probability:.2%}")
    print("\nNote: This is a learning project. Predictions are estimates, not historical certainty.")


if __name__ == "__main__":
    main()
