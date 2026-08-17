# ML Assignment 2 - Classification Model Comparison

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

<table>
  <thead>
    <tr>
      <th>ML Model Name</th>
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9825</td>
      <td>0.9954</td>
      <td>0.9861</td>
      <td>0.9861</td>
      <td>0.9861</td>
      <td>0.9623</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>0.9035</td>
      <td>0.9358</td>
      <td>0.9420</td>
      <td>0.9028</td>
      <td>0.9220</td>
      <td>0.7969</td>
    </tr>
    <tr>
      <td>kNN</td>
      <td>0.9737</td>
      <td>0.9884</td>
      <td>0.9600</td>
      <td>1.0000</td>
      <td>0.9796</td>
      <td>0.9442</td>
    </tr>
    <tr>
      <td>Naive Bayes</td>
      <td>0.9386</td>
      <td>0.9878</td>
      <td>0.9452</td>
      <td>0.9583</td>
      <td>0.9517</td>
      <td>0.8676</td>
    </tr>
    <tr>
      <td>Random Forest (Ensemble)</td>
      <td>0.9561</td>
      <td>0.9944</td>
      <td>0.9589</td>
      <td>0.9722</td>
      <td>0.9655</td>
      <td>0.9054</td>
    </tr>
  </tbody>
</table>

## Observations About Model Performance

<table>
  <thead>
    <tr>
      <th>ML Model Name</th>
      <th>Observation about model performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic Regression</td>
      <td>Usually performs very well on this dataset because the features are informative and mostly separable after scaling.</td>
    </tr>
    <tr>
      <td>Decision Tree</td>
      <td>Easy to interpret, but can overfit if the tree is allowed to grow too deep. A controlled depth improves generalization.</td>
    </tr>
    <tr>
      <td>kNN</td>
      <td>Performs well when features are scaled because distance-based learning is sensitive to feature magnitude.</td>
    </tr>
    <tr>
      <td>Naive Bayes</td>
      <td>Fast and simple. It can perform strongly, but its independence assumption may limit performance on correlated medical features.</td>
    </tr>
    <tr>
      <td>Random Forest (Ensemble)</td>
      <td>Generally robust because it combines many decision trees and reduces the overfitting risk of a single tree.</td>
    </tr>
    <tr>
      <td>Overall Winner</td>
      <td>Logistic Regression is the overall winner because it has the highest Accuracy, AUC, F1, and MCC on the test split.</td>
    </tr>
  </tbody>
</table>

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
