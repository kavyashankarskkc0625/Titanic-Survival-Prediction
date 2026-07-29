# 🚢 Titanic Survival Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-blue)

## 📌 Project Overview

This project is my first end-to-end Machine Learning project using the famous **Titanic - Machine Learning from Disaster** competition on Kaggle.

The objective is to build a Machine Learning model that predicts whether a passenger survived the Titanic disaster based on passenger information such as:

- Gender
- Age
- Passenger Class
- Fare
- Family Size
- Embarkation Port

This project covers the complete Machine Learning workflow from loading the dataset to generating predictions for Kaggle submission.

---

# 📚 Table of Contents

1. Problem Statement
2. Dataset Information
3. Project Structure
4. Technologies Used
5. Step 1 - Loading the Dataset
6. Step 2 - Exploratory Data Analysis (EDA)
7. Step 3 - Data Cleaning
8. Step 4 - Feature Engineering & Encoding
9. Step 5 - Preparing Features and Target
10. Step 6 - Train-Test Split
11. Step 7 - Model Building
12. Step 8 - Model Evaluation
13. Step 9 - Feature Importance
14. Step 10 - Making Predictions
15. Step 11 - Kaggle Submission
16. Results
17. Future Improvements
18. How to Run the Project

---

# 🎯 Problem Statement

The RMS Titanic sank on April 15, 1912 after hitting an iceberg.

Not everyone survived.

Given passenger information, the task is to predict whether a passenger survived.

This is a **Binary Classification Problem**.

Output:

- 0 → Did Not Survive
- 1 → Survived

---

# 📂 Dataset Information

The competition provides two datasets.

## train.csv

Contains passenger information along with the actual survival values.

Example:

| PassengerId | Sex | Age | Pclass | Survived |
|-------------|------|------|---------|------------|
|1|Male|22|3|0|
|2|Female|38|1|1|

This dataset is used to train the model.

---

## test.csv

Contains passenger information only.

There is **no Survived column** because the model has to predict it.

---

# 📁 Project Structure

```
Titanic/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── notebooks/
│   └── titanic.ipynb
│
├── src/
│
├── submission.csv
│
├── requirements.txt
│
└── README.md
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook
- VS Code

---

# 📥 Step 1 - Loading the Dataset

```python
import pandas as pd

train = pd.read_csv("../data/train.csv")
```

Check the first few rows.

```python
train.head()
```

Check dataset information.

```python
train.info()
```

Check statistical summary.

```python
train.describe()
```

---

# 🔍 Step 2 - Exploratory Data Analysis (EDA)

EDA helps us understand the data before building the model.

Questions explored:

- How many passengers survived?
- Survival percentage
- Survival by Gender
- Survival by Passenger Class
- Survival by Embarked Port
- Age distribution
- Missing values

---

## Survival Count

```python
train["Survived"].value_counts()
```

Output

```
0    549
1    342
```

Approximately **38%** passengers survived.

---

## Survival Percentage

```python
train["Survived"].value_counts(normalize=True)
```

Output

```
Did Not Survive → 61.6%

Survived → 38.4%
```

---

## Survival by Gender

```python
train.groupby("Sex")["Survived"].mean()
```

Result

| Gender | Survival Rate |
|----------|--------------|
| Female | 74% |
| Male | 19% |

Observation:

Women had a much higher survival rate.

---

## Survival by Passenger Class

```python
train.groupby("Pclass")["Survived"].mean()
```

Result

| Class | Survival Rate |
|---------|--------------|
|1|63%|
|2|47%|
|3|24%|

Observation:

First-class passengers had the highest survival rate.

---

## Survival by Embarked Port

```python
train.groupby("Embarked")["Survived"].mean()
```

Observation:

Passengers boarding at Cherbourg had higher survival rates.

---

## Age Distribution

```python
train["Age"].hist()
```

Observation:

Most passengers were between **20-40 years old**.

---

## Missing Values

```python
train.isnull().sum()
```

Result

| Column | Missing |
|-----------|---------|
|Age|177|
|Cabin|687|
|Embarked|2|

Observation:

Cabin had too many missing values.

---

# 🧹 Step 3 - Data Cleaning

## Handling Missing Age

Median was used because it is less affected by outliers.

```python
train["Age"] = train["Age"].fillna(train["Age"].median())
```

---

## Handling Missing Embarked

```python
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
```

---

## Removing Cabin

Cabin had nearly 77% missing values.

```python
train = train.drop(columns=["Cabin"])
```

---

# ⚙ Step 4 - Feature Engineering & Encoding

Machine Learning models work with numerical values.

---

## Encoding Gender

```python
train["Sex"] = train["Sex"].map({
    "male":0,
    "female":1
})
```

Result

| Original | Encoded |
|-----------|----------|
|Male|0|
|Female|1|

---

## One Hot Encoding Embarked

```python
train = pd.get_dummies(
    train,
    columns=["Embarked"],
    drop_first=True
)
```

Generated columns

```
Embarked_Q

Embarked_S
```

---

## Dropping Unnecessary Columns

These columns were removed because they do not directly contribute to prediction.

```python
train = train.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket"
    ]
)
```

---

# 🎯 Step 5 - Preparing Features and Target

Features (X)

Everything except the answer.

```python
X = train.drop(columns=["Survived"])
```

Target (y)

The value we want to predict.

```python
y = train["Survived"]
```

---

# ✂ Step 6 - Train Test Split

To evaluate the model fairly, the dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Output

```
Training Samples : 712

Testing Samples : 179
```

---

# 🌳 Step 7 - Building the Machine Learning Model

Decision Tree Classifier was selected as the baseline model.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)
```

Train the model.

```python
model.fit(
    X_train,
    y_train
)
```

The model learns patterns from the training data.

Examples:

- Females had higher survival.
- First-class passengers survived more often.
- Higher fares often indicated better survival chances.

---

# 📈 Step 8 - Model Evaluation

Generate predictions.

```python
predictions = model.predict(X_test)
```

Calculate accuracy.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)
```

Result

```
Accuracy = 78.77%
```

---

# ⭐ Step 9 - Feature Importance

Feature importance indicates which variables influenced the model the most.

| Feature | Importance |
|----------|------------|
|Sex|0.310|
|Fare|0.264|
|Age|0.206|
|Pclass|0.112|
|SibSp|0.060|
|Parch|0.027|
|Embarked_S|0.016|
|Embarked_Q|0.005|

Observation:

Gender was the strongest predictor of survival.

---

# 🚀 Step 10 - Predicting Test Dataset

Load test dataset.

```python
test = pd.read_csv("../data/test.csv")
```

Apply the same preprocessing steps used for training.

- Fill missing Age
- Fill missing Fare
- Drop Cabin
- Encode Gender
- One Hot Encode Embarked
- Remove PassengerId, Name and Ticket

Predict.

```python
predictions = model.predict(test)
```

---

# 📤 Step 11 - Creating Kaggle Submission

Create submission dataframe.

```python
submission = pd.DataFrame({
    "PassengerId": test_original["PassengerId"],
    "Survived": predictions
})
```

Save.

```python
submission.to_csv(
    "submission.csv",
    index=False
)
```

Upload the generated file to Kaggle.

---

# 📊 Results

Model Used

- Decision Tree Classifier

Validation Accuracy

```
78.77%
```

The model successfully generated predictions for the Kaggle competition.

---

# 🚀 Future Improvements

This baseline model can be improved by:

- Feature Engineering
  - Family Size
  - IsAlone
  - Passenger Title (Mr, Mrs, Miss)
- Hyperparameter Tuning
- Cross Validation
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Expected accuracy after improvements:

```
82% - 86%
```

---

# ▶ How to Run

Clone the repository.

```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run the notebook.

```bash
jupyter notebook
```

Execute all cells.

Generate `submission.csv`.

Upload it to Kaggle.

---

# 📖 Key Learnings

During this project, I learned:

- Understanding a Machine Learning problem
- Exploratory Data Analysis (EDA)
- Handling missing values
- Encoding categorical variables
- Preparing features and target
- Splitting data into training and testing sets
- Building a Decision Tree classifier
- Evaluating model performance
- Understanding feature importance
- Generating predictions on unseen data
- Creating a Kaggle submission

This project provided hands-on experience with the complete Machine Learning workflow and served as a strong foundation for more advanced machine learning models and Kaggle competitions.

---

## 👨‍💻 Author

**Kavya S**

- GitHub: https://github.com/kavyashankarskkc0625
- LinkedIn: https://linkedin.com/in/kavyashankar25s

---
⭐ If you found this project helpful, feel free to star the repository!