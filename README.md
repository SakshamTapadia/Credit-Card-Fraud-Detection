# Credit Card Fraud Detection

## Project Overview

This project aims to detect fraudulent credit card transactions using advanced machine learning techniques. The dataset contains a large number of credit card transactions, including both fraudulent and non-fraudulent cases. The goal is to build a predictive model that can accurately identify fraudulent transactions.

## Dataset

The dataset used in this project can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download). It contains the following columns:

- **Time**: The amount of time elapsed since the first transaction in the dataset.
- **V1** to **V28**: The principal components obtained with PCA, which are anonymized features.
- **Amount**: The transaction amount.
- **Class**: The class label (0 for non-fraudulent, 1 for fraudulent).

## Project Notebook

The complete project notebook is available [here](https://github.com/SakshamTapadia/CODSOFT/blob/main/Task5%20-Credit%20Card%20Fraud%20Detection.ipynb). It includes the following steps:

1. **Data Exploration and Preprocessing**:
    - Load and inspect the dataset.
    - Handle missing values.
    - Normalize the 'Amount' feature.
    - Split the dataset into training and testing sets.

2. **Exploratory Data Analysis (EDA)**:
    - Visualize the distribution of fraudulent and non-fraudulent transactions.
    - Analyze the distribution of the 'Amount' and 'Time' features.
    - Examine the correlation between features.

3. **Model Building**:
    - Address class imbalance using techniques such as undersampling, oversampling, or SMOTE.
    - Train multiple machine learning models (e.g., Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
    - Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

4. **Model Evaluation and Selection**:
    - Compare the performance of different models.
    - Select the best model based on evaluation metrics.

5. **Prediction and Conclusion**:
    - Make predictions on the test set.
    - Summarize the findings and conclusion.

## Results

The results of the model evaluation and the final predictions on the test set are included in the project notebook. The best performing model achieved high accuracy and provided insights into the most important features influencing fraud detection.

## Conclusion

This project demonstrates the application of advanced machine learning techniques to detect fraudulent credit card transactions. By leveraging data preprocessing, exploratory data analysis, and model evaluation, we can build accurate predictive models and gain valuable insights into fraud detection.
