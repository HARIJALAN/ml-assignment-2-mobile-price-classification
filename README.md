# Mobile Price Classification – ML Assignment 2

## 1. Problem Statement
The objective of this project is to build, evaluate, and deploy multiple machine
learning classification models to predict the price range of mobile phones based
on their hardware specifications.

The project demonstrates an end-to-end machine learning workflow including data
acquisition, model training, evaluation using multiple metrics, and deployment
through an interactive Streamlit web application.

---

## 2. Dataset Description
The Mobile Price Classification dataset is obtained from Kaggle and downloaded
programmatically using the **kagglehub** library. The dataset contains information
about mobile phone specifications such as battery power, RAM, internal memory,
screen dimensions, camera features, and connectivity options.

- **Number of instances:** 2000  
- **Number of features:** 20 (all numerical)  
- **Target variable:** `price_range`  

The target variable categorizes mobile phones into four price classes:
- **0** → Low Cost  
- **1** → Medium Cost  
- **2** → High Cost  
- **3** → Very High Cost  

The dataset contains no missing values and is suitable for multi-class
classification tasks.

---

## 3. Models Used and Evaluation Metrics

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model was evaluated using the following metrics:
- Accuracy  
- AUC Score (One-vs-Rest)  
- Precision (Weighted)  
- Recall (Weighted)  
- F1 Score (Weighted)  
- Matthews Correlation Coefficient (MCC)  

---

### 3.1 Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8225 | 0.9445 | 0.8162 | 0.8225 | 0.8181 | 0.7641 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| KNN | 0.5000 | 0.7698 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Random Forest | 0.8800 | 0.9767 | 0.8796 | 0.8800 | 0.8797 | 0.8400 |
| XGBoost | **0.9350** | **0.9945** | **0.9355** | **0.9350** | **0.9350** | **0.9135** |

---

## 4. Observations on Model Performance

| Model | Observation |
|------|------------|
| Logistic Regression | Provides a strong baseline after feature scaling with stable performance |
| Decision Tree | Easy to interpret but slightly prone to overfitting |
| KNN | Shows poor performance due to sensitivity to feature space and class overlap |
| Naive Bayes | Efficient and fast, performing well despite independence assumptions |
| Random Forest | Robust model with strong generalization capability |
| XGBoost | Achieves the best overall performance across all evaluation metrics |

---

## 5. Streamlit Application Features
An interactive Streamlit web application was developed and deployed with the
following features:

- Dataset upload option (CSV format for test data)
- Model selection dropdown to choose among multiple trained models
- Display of evaluation metrics for the selected model
- Visualization of confusion matrix
- Download option to export evaluation metrics of all models as a CSV file

The download feature enables evaluators and users to verify and compare model
performance results offline.

---

## 6. Deployment
The Streamlit application was deployed using **Streamlit Community Cloud**.

The model training and evaluation were performed on the **BITS Virtual Lab**
environment, and a screenshot of execution has been included in the submission
PDF as proof of compliance.

The live application link is provided as part of the assignment submission.

---

## 7. Conclusion
This project successfully demonstrates the complete lifecycle of a machine
learning classification task, from dataset acquisition and model development to
evaluation and deployment using an interactive web interface. Ensemble models,
especially **XGBoost**, showed superior performance on the Mobile Price
Classification dataset.
