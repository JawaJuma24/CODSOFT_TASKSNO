# CodSoft Data Science Internship — Task 1
## Titanic Survival Prediction

### Objective
Build a beginner-friendly machine-learning model that predicts whether a Titanic passenger survived using passenger details.

### Tools
- Python
- pandas
- scikit-learn
- Logistic Regression

### Features used
- Passenger class (`Pclass`)
- Sex
- Age
- Number of siblings/spouses aboard (`SibSp`)
- Number of parents/children aboard (`Parch`)
- Fare
- Port of embarkation (`Embarked`)

### Project setup
1. Install Python 3.
2. Download the Titanic training dataset from Kaggle: https://www.kaggle.com/competitions/titanic/data
3. Put the downloaded `train.csv` file in the same folder as `main.py`.
4. Open Terminal in this folder.
5. Create and activate a virtual environment (recommended):

   macOS:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

6. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
7. Run:
   ```bash
   python main.py
   ```

### What the program does
1. Loads the dataset.
2. Selects relevant features and the `Survived` target.
3. Fills missing numeric and categorical values.
4. Scales numeric data and encodes categories.
5. Splits data into training and testing sets.
6. Trains a Logistic Regression model.
7. Reports accuracy, precision/recall/F1, and a confusion matrix.
8. Predicts survival for one hypothetical passenger.

### Important
- Use Kaggle's `train.csv`, because it contains the `Survived` column.
- Do not invent an accuracy number for your LinkedIn post. Run the program and report the result it actually prints.
- This is a learning demonstration, not a perfect predictor of individual historical outcomes.

### Suggested GitHub repository
`CODSOFT_TASKSNO`

You can place this project in a `Task1_Titanic_Survival_Prediction` folder inside the repository.
