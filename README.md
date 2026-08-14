# Megaline Plan Classification

Machine learning project developed to classify Megaline customers into the **Smart** or **Ultra** mobile plans based on their monthly usage behavior.

## Project Overview

Megaline wants to encourage customers who still use legacy mobile plans to migrate to one of its newer plans: **Smart** or **Ultra**.

The objective of this project is to build a classification model capable of analyzing customer behavior and predicting which of the two plans is more appropriate.

The minimum required accuracy for the project was **0.75** on the test set.

## Dataset

The dataset contains **3,214 monthly customer observations** with information about mobile service usage.

### Features

- `calls` — number of calls made
- `minutes` — total call duration in minutes
- `messages` — number of text messages sent
- `mb_used` — internet traffic used in megabytes

### Target

- `is_ultra`
  - `0` — Smart plan
  - `1` — Ultra plan

The dataset had already been preprocessed before this project, allowing the analysis to focus directly on model development and evaluation.

## Methodology

The project follows a supervised machine learning classification workflow:

1. Load and inspect the dataset
2. Separate features and target
3. Split the data into:
   - 60% training
   - 20% validation
   - 20% testing
4. Train and compare multiple classification models
5. Tune selected hyperparameters
6. Select the best-performing model using validation accuracy
7. Evaluate the selected model on the test set
8. Compare the final model against a simple baseline classifier

## Models Evaluated

Three classification algorithms were evaluated:

- Decision Tree
- Random Forest
- Logistic Regression

### Validation Results

| Model | Best Configuration | Validation Accuracy |
| --- | --- | ---: |
| Random Forest | `n_estimators=40`, `max_depth=8` | **0.8087** |
| Decision Tree | `max_depth=3` | 0.7854 |
| Logistic Regression | `solver='liblinear'` | 0.7574 |

The **Random Forest Classifier** achieved the highest validation accuracy and was selected as the final model.

## Final Model

The selected Random Forest model used:

```python
RandomForestClassifier(
    n_estimators=40,
    max_depth=8,
    random_state=12345
)
```

### Test Performance

| Metric | Score |
| --- | ---: |
| Test Accuracy | **0.7963** |
| Required Accuracy | 0.7500 |
| Dummy Classifier Accuracy | 0.6843 |

The final model exceeded both the minimum project requirement and the baseline classifier.

## Key Findings

- Random Forest achieved the strongest validation performance among the tested models.
- Hyperparameter tuning improved the ability of tree-based models to generalize.
- The final model reached approximately **79.6% accuracy** on previously unseen test data.
- The model outperformed a naive classifier that always predicts the majority class.
- The sanity check indicates that the model learned useful behavioral patterns rather than simply exploiting the class distribution.

## Technologies

- Python
- pandas
- scikit-learn
- Jupyter Notebook

## Repository Structure

```text
megaline-plan-classification/
│
├── data/
│   └── users_behavior.csv
│
├── megaline-plan-classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## How to Run

1. Clone this repository.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Open `megaline-plan-classification.ipynb` in Jupyter Notebook, JupyterLab, or Visual Studio Code.

4. Run all notebook cells sequentially.

The dataset is included in the `data/` directory.

## Dataset Source

The dataset was provided as part of the **TripleTen Data Science Bootcamp**.

Original dataset:

https://practicum-content.s3.us-west-1.amazonaws.com/datasets/users_behavior.csv

## About This Project

This project was originally developed as part of the **TripleTen Data Science Bootcamp** and was later organized and documented for inclusion in my professional data science portfolio.