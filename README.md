# ML Assignment 2 - Classification Model Comparison

## Student Details

- Name: CHITRA V
- BITS ID: 2025ac05483

## Problem Statement

The objective of this project is to build, evaluate, compare, and deploy multiple machine learning classification models on a single public classification dataset. The final solution includes trained models, test data, a Streamlit web app, and model performance observations.

## Dataset Description

This project uses the Breast Cancer Wisconsin Diagnostic dataset available through `scikit-learn`, originally from the UCI Machine Learning Repository.

- Problem type: Binary classification
- Number of instances: 569
- Number of features: 30
- Target classes: malignant and benign
- Target column: `target`

The dataset satisfies the assignment requirement of at least 500 instances and at least 12 features.

## GitHub Repository Link

[GitHub Repository](https://github.com/2025ac05483/ML_Assignment2)

## Live Streamlit App Link

[Live Streamlit App](https://mlassignment2-5xuqec6rhhj3smgxcensvw.streamlit.app/)

## Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

Note: The assignment PDF mentions six models in one place, but lists five specific models. This project implements all five models explicitly listed in the assignment.

## Model Comparison Table

The following results were generated using an 80:20 stratified train-test split with `random_state=42`.

```bash
python3 train_models.py
```

The generated comparison table is also saved at `model/metrics.csv`.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9358 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

## Observations About Model Performance

| Model | Observation |
| --- | --- |
| Logistic Regression | Best overall result; features separate well after scaling. |
| Decision Tree | Interpretable, but weaker than the other models on this split. |
| kNN | Strong performance after scaling, with perfect recall. |
| Naive Bayes | Fast and simple, but affected by correlated medical features. |
| Random Forest | Robust ensemble model with balanced performance. |
| Overall Winner | Logistic Regression, because it has the highest Accuracy, AUC, F1, and MCC. |

## How To Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models and generate artifacts:

```bash
python3 train_models.py
```

Run Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
project-folder/
  app.py
  train_models.py
  requirements.txt
  README.md
  test_data.csv
  model/
    metadata.joblib
    metrics.csv
    *.joblib
```

## Streamlit App Features

- CSV upload option for test data
- Model selection dropdown
- Evaluation metric display
- Confusion matrix
- Classification report
- Prediction table

## Final Submission Checklist

- GitHub repository link works
- Streamlit app link opens correctly
- App loads without errors
- All required features are implemented
- README content is copied into the submitted PDF
- BITS Virtual Lab execution screenshot is included
