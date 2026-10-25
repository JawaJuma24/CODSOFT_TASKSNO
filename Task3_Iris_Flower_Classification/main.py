"""CodSoft Data Science Internship - Task 3: Iris Flower Classification."""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("=" * 48)
    print("CODSOFT TASK 3: IRIS FLOWER CLASSIFICATION")
    print("=" * 48)
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {', '.join(X.columns)}")
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=[0, 1, 2],
        target_names=iris.target_names, zero_division=0
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
    predicted_class = model.predict(sample)[0]
    print("\nExample flower measurements:")
    print(sample.to_string(index=False))
    print(f"Predicted species: {iris.target_names[predicted_class]}")
    print("\nNote: The score is specific to this train/test split.")


if __name__ == "__main__":
    main()
