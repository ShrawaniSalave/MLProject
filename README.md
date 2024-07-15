# MLProject
# Title: Celiac Disease Prediction Using Machine Learning
## Introduction
Celiac disease is an autoimmune disorder triggered by gluten, causing digestive issues. Early detection is crucial for effective management. This project aims to develop a robust ML model to predict celiac disease risk, improving diagnostic accuracy and facilitating proactive management

## Data Collection and Processing
The dataset, sourced from Kaggle, includes genetic markers, clinical symptoms, and demographic information. Preprocessing steps, including handling missing values with dropna(), ensured data quality.

## Feature Selection
Key features were selected using Correlation Heatmap Analysis and Information Gain Calculation to identify the most informative features, reducing redundancy and enhancing predictive accuracy

## Model Selection
Various models were evaluated:
## Model Selection

| Model                | Accuracy (%) | Confusion Matrix     |
|----------------------|--------------|----------------------|
| Random Forest        | 95           | [[ 92, 26], [ 4, 540]] |
| SVM                  | 93           | [[ 44, 26], [ 3, 369]] |
| Naive Bayes          | 88.97        | [[ 73, 45], [ 28, 516]] |
| Logistic Regression  | 86.65        | [[ 37, 32], [ 12, 361]] |


## Model Description
1.Import Libraries
2.Read Dataset
3.Data Cleaning (dropna())
4.Data Visualization
5.Feature Selection
6.Split Dataset (70:30)
7.Data Normalization
8.Model Training and Testing
9.Model Evaluation
10.Tree Visualization and Optimization

## Testing and Evaluation
The model was assessed using accuracy, precision, recall, F1-score, and confusion matrix metrics from scikit-learn.

## Results
The Random Forest model achieved 95.46% accuracy. Key features included Age, Abdominal Symptoms, Short Stature, Weight Loss, IgA Levels, and IgG Levels. The optimized model maintained high accuracy (92.44%), demonstrating stability and reliability.

## Conclusion
Our ML model for celiac disease prediction achieved 95% accuracy, offering significant insights into key diagnostic factors. This approach enhances clinical decision-making and lays the foundation for advancements in personalized medicine and targeted interventions.


