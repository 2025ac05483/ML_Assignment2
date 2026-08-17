# Step-by-Step Assignment Explanation

## Step 1: Choose The Dataset

I used the Breast Cancer Wisconsin Diagnostic dataset from `scikit-learn`, originally from the UCI Machine Learning Repository.

Why this dataset is suitable:

- It is a classification dataset.
- It has 569 instances, which is more than the required 500.
- It has 30 input features, which is more than the required 12.
- It has two classes: malignant and benign.

In the code, the dataset is loaded in `train_models.py` using:

```python
from sklearn.datasets import load_breast_cancer
```

The target column is named `target`. The remaining 30 columns are the input features.

## Step 2: Split The Dataset

The dataset is split into training and testing data.

- Training data: 80%
- Testing data: 20%

The test split is stratified, which means the class ratio is preserved in both training and testing sets. This is important for classification problems because it avoids creating an unfair test set.

The test data is saved as:

```text
test_data.csv
```

This file is required by the assignment and is also used by the Streamlit app.

## Step 3: Train The Required Models

The assignment asks for the listed classification models. This project trains:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

Logistic Regression and kNN use `StandardScaler` because they are sensitive to feature scale.

Decision Tree, Naive Bayes, and Random Forest do not require scaling in the same way.

Each trained model is saved inside the `model/` folder as a `.joblib` file.

## Step 4: Evaluate The Models

Each model is evaluated using the six metrics required by the assignment:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC Score

The generated results are saved in:

```text
model/metrics.csv
```

Current results:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9358 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

Logistic Regression is the winner on this split because it has the highest Accuracy, AUC, F1, and MCC.

## Step 5: Build The Streamlit App

The app is implemented in:

```text
app.py
```

The app includes the assignment-required features:

- CSV upload option
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix
- Classification report
- Prediction table

If the uploaded CSV contains the `target` column, the app calculates metrics. If the uploaded CSV does not contain `target`, the app only shows predictions.

## Step 6: Create requirements.txt

The `requirements.txt` file lists all packages needed by Streamlit Cloud:

```text
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
```

This is important because Streamlit Community Cloud installs dependencies from this file.

## Step 7: Upload To GitHub

Create a GitHub repository and upload these files:

```text
app.py
train_models.py
requirements.txt
README.md
test_data.csv
model/
```

The `model/` folder must include the saved `.joblib` files, `metrics.csv`, and `metadata.joblib`.

## Step 8: Deploy On Streamlit Community Cloud

After pushing to GitHub:

1. Go to `https://streamlit.io/cloud`
2. Sign in with GitHub
3. Click New App
4. Select your repository
5. Select the main branch
6. Set the app file as `app.py`
7. Click Deploy

After deployment, copy the Streamlit app link into your README and final PDF.

## Step 9: BITS Virtual Lab Screenshot

The assignment requires one screenshot proving execution on BITS Virtual Lab.

Run the project on BITS Virtual Lab and take a screenshot showing either:

- `python3 train_models.py` execution, or
- the Streamlit app running in the browser

Add that screenshot to the final PDF.

## Step 10: Final PDF Submission

Your final PDF must contain, in this order:

1. GitHub repository link
2. Live Streamlit app link
3. BITS Virtual Lab execution screenshot
4. README content

The README content from `README.md` can be copied directly into the PDF after you replace the placeholder GitHub and Streamlit links.
