# CodSoft Data Science Internship — Task 2
## Movie Rating Prediction with Python

### Objective
Build a regression model that estimates a movie's rating using details such as genre, director, actors, year, duration, and votes.

### Tools
- Python
- pandas
- scikit-learn
- Random Forest Regressor

### Dataset
This starter project is designed for the common IMDb Movies India dataset, which typically contains columns such as `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`, `Director`, `Actor 1`, `Actor 2`, and `Actor 3`.

Dataset page: https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies

1. Download and extract the dataset.
2. Find the movie ratings CSV file.
3. Rename it to `IMDb Movies India.csv`.
4. Place it in the same folder as `main.py`.

If your file has a different name, change `DATA_FILE` near the top of `main.py`.

### Run on macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

### Workflow
1. Load the dataset.
2. Clean rating, year, duration, and vote values.
3. Select available features.
4. Fill missing values and encode categories.
5. Split into training and test sets.
6. Train a Random Forest regression model.
7. Evaluate using MAE, RMSE, and R-squared.
8. Compare actual and predicted ratings for test examples.

### Metrics
- **MAE:** Average absolute prediction error in rating points. Lower is better.
- **RMSE:** Gives larger errors more weight. Lower is better.
- **R²:** Measures how well predictions explain variation in ratings. Closer to 1 is generally better; it can be negative.

### Important
- Do not claim a specific score until you have run the program.
- Predictions are estimates, not guaranteed ratings.
- Check the dataset license and terms before redistributing it. You can upload your code and README without uploading the CSV.
