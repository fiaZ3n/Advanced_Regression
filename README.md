# House Prices - Advanced Regression Techniques

A machine learning regression project based on Kaggle's **House Prices - Advanced Regression Techniques** competition.

## Project Overview

The goal is to predict the **SalePrice** of residential homes in Ames, Iowa using property features. The dataset contains 79 explanatory variables describing different aspects of each house. 

This project practices:
- Data preprocessing
- Missing-value handling
- Categorical feature encoding
- Regression modeling
- Model evaluation
- Final price prediction

## Dataset

The project uses:
- `train.csv` — training features + `SalePrice`
- `test.csv` — test features for final prediction
- `Id` — house identifier

Kaggle requires the final submission to contain `Id` and `SalePrice`.

## Workflow

```text
Load train.csv + test.csv
        ↓
Combine datasets
        ↓
Set Id as index
        ↓
Inspect missing values
        ↓
Remove columns with excessive missing values
        ↓
Handle categorical features
        ↓
One-hot encoding
        ↓
Fill remaining numerical missing values
        ↓
Separate training/testing data
        ↓
Create X and y
        ↓
Train/validation split
        ↓
Train regression models
        ├── Linear Regression
        ├── XGBoost
        └── Random Forest
        ↓
Evaluate models
        ↓
Retrain selected model on full training data
        ↓
Predict test data
        ↓
Generate output.csv
```

## Data Preprocessing

### 1. Load and combine data

`train.csv` and `test.csv` are loaded with pandas and temporarily combined so preprocessing and categorical encoding are consistent.

### 2. Missing-value analysis

Missing values are counted and visualized with a heatmap.

### 3. Remove columns with excessive missing values

Categorical columns with more than 1100 missing values are removed.

### 4. Categorical encoding

Categorical values are temporarily filled with `"null"` and converted to dummy variables using:

```python
pd.get_dummies()
```

Dummy columns containing `"null"` are then removed.

### 5. Numerical missing values

Remaining missing numerical values are filled using either:
- Mode for selected columns
- Mean for selected columns

## Models

### Linear Regression

```python
model_1 = LinearRegression()
```

Used as a basic regression baseline.

### XGBoost

```python
model_2 = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    random_state=42
)
```

Used as a gradient-boosting regression model.

### Random Forest

```python
model_3 = RandomForestRegressor(
    n_estimators=1000,
    random_state=42
)
```

Used as an ensemble tree-based regression model.

## Model Evaluation

The training data is split into training and validation sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

Models are evaluated on the validation data without training on it.

Example:

```python
model_2.fit(X_train, Y_train)
y_pred = model_2.predict(X_test)

mean_squared_error(Y_test, y_pred)
```

## Final Prediction

After evaluating the models, the selected model is retrained using all available training data:

```python
model_2.fit(X, y)
```

Then predictions are generated:

```python
pred = model_2.predict(testing_data)
```

## Generate `output.csv`

```python
final = pd.DataFrame()
final['Id'] = testing_data.index
final['SalePrice'] = pred

final.to_csv('output.csv', index=False)
```

This creates:

```text
output.csv
```

with:

```text
Id,SalePrice
```

## Kaggle Evaluation

Kaggle evaluates submissions using **Root Mean Squared Error (RMSE)** between the logarithm of predicted and observed sale prices.

Therefore, the MSE used in this notebook for model comparison is a learning/validation metric and is **not the same as Kaggle's official leaderboard metric**.

## Key Lessons

- How to inspect and handle missing values
- How to encode categorical features
- How to split data into training and validation sets
- How regression models are trained and evaluated
- How XGBoost and Random Forest work as regression models
- How to retrain a selected model on the full dataset
- How to generate a Kaggle submission file
- Why validation data should not be used during model training

## Project Structure

```text
DataScience/
│
├── train.csv
├── test.csv
├── Advanced_regression.ipynb
└── output.csv
```

## Competition

**Kaggle — House Prices: Advanced Regression Techniques**

The competition focuses on predicting residential sale prices and practicing feature engineering, random forests, and gradient boosting.
