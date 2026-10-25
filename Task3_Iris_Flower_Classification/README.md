# CodSoft Data Science Internship — Task 3
## Iris Flower Classification

### Objective
Train a machine-learning model to classify an iris flower into one of three species—setosa, versicolor, or virginica—using four measurements: sepal length, sepal width, petal length, and petal width.

### Tools
- Python
- pandas
- scikit-learn
- K-Nearest Neighbors (KNN)

### Dataset
This project uses the Iris dataset built into scikit-learn. No separate CSV download is needed. It contains 150 samples and four numeric features.

### Run on macOS in VS Code
Open this folder in VS Code and run these commands in the terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

If `.venv` already exists in this folder, skip the first command.

### Workflow
1. Load the built-in Iris dataset.
2. Split the data into training and testing sets.
3. Standardize numeric features.
4. Train a K-Nearest Neighbors classifier.
5. Predict species for test samples.
6. Evaluate using accuracy, a classification report, and a confusion matrix.
7. Predict the species for an example flower.

### Understanding the output
- **Accuracy:** Fraction of test examples classified correctly.
- **Precision:** How many predictions for a species were correct.
- **Recall:** How many real examples of a species were found.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Rows are actual classes; columns are predicted classes.

Do not claim a score until you run the program. Results depend on the train/test split and model settings.
