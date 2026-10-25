"""CodSoft Data Science Internship - Task 2: Movie Rating Prediction."""
from pathlib import Path
import re
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path("IMDb Movies India.csv")

ALIASES = {
    "name": "Name", "movie": "Name", "movie name": "Name", "title": "Name",
    "year": "Year", "release year": "Year", "duration": "Duration", "runtime": "Duration",
    "genre": "Genre", "genres": "Genre", "rating": "Rating", "imdb rating": "Rating",
    "imdb_rating": "Rating", "votes": "Votes", "vote": "Votes",
    "director": "Director", "director name": "Director",
    "actor 1": "Actor 1", "actor1": "Actor 1", "actor 2": "Actor 2",
    "actor2": "Actor 2", "actor 3": "Actor 3", "actor3": "Actor 3",
}

def normalize_column_name(column):
    cleaned = re.sub(r"\s+", " ", str(column).strip().replace("_", " ")).lower()
    return ALIASES.get(cleaned, str(column).strip())

def to_numeric_series(series):
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path.name}'. Download the IMDb Movies India CSV, "
            "extract it, and place it beside main.py. If the filename differs, "
            "change DATA_FILE near the top of main.py."
        )
    data = pd.read_csv(path, encoding="latin1")
    data.columns = [normalize_column_name(c) for c in data.columns]
    if data.empty:
        raise ValueError("The dataset is empty.")
    if "Rating" not in data.columns:
        raise ValueError(f"Could not find a Rating column. Columns found: {list(data.columns)}")
    return data

def main():
    try:
        data = load_dataset(DATA_FILE)
    except Exception as error:
        print(f"\nDataset error: {error}")
        sys.exit(1)

    possible_features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3",
                         "Year", "Duration", "Votes"]
    features = [c for c in possible_features if c in data.columns]
    if not features:
        print("No supported feature columns found. Expected genre, director, actors, year, duration, or votes.")
        print("Columns found:", list(data.columns))
        sys.exit(1)

    data["Rating"] = to_numeric_series(data["Rating"])
    data = data.dropna(subset=["Rating"]).copy()
    numeric_features = [c for c in ["Year", "Duration", "Votes"] if c in features]
    categorical_features = [c for c in features if c not in numeric_features]

    for c in numeric_features:
        data[c] = to_numeric_series(data[c])
    for c in categorical_features:
        data[c] = data[c].replace(["", " ", "nan", "NaN", "None", "unknown", "Unknown"], pd.NA)
    data = data.dropna(subset=features, how="all")

    if len(data) < 20:
        print(f"Only {len(data)} usable rows found. Please use a larger movie dataset.")
        sys.exit(1)

    X, y = data[features], data["Rating"]
    transformers = []
    if numeric_features:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_features))

    preprocessing = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=2
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\n=== MOVIE RATING PREDICTION ===")
    print(f"Usable movies: {len(data)}")
    print("Features used:", ", ".join(features))
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print("\nTraining model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    print("\n=== MODEL EVALUATION ===")
    print(f"Mean Absolute Error (MAE): {mae:.3f} rating points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} rating points")
    print(f"R-squared (R²): {r2:.3f}")
    print("\n=== SAMPLE TEST PREDICTIONS ===")
    print(pd.DataFrame({
        "Actual Rating": y_test.iloc[:5].to_numpy(),
        "Predicted Rating": predictions[:5].round(2),
    }).to_string(index=False))
    print("\nPredictions are estimates. Metrics depend on the dataset and selected features.")

if __name__ == "__main__":
    main()
