# Diabetes Risk Predictor
## Problem
In India millions have prediabetes and don't know it. This app lets them enter basic health values and check their risk
## Dataset
Dataset contains 768 patients with 8 health features including glucose, BMI, insulin and more. Originally from UCI Machine Learning Repository, accessed via Kaggle
## Approach
* Loaded and explored data using Pandas — checked statistics, missing values, distributions
* Performed EDA using matplotlib and seaborn — glucose distribution, correlation heatmap
* Built Logistic Regression model with 80/20 train test split and feature scaling
* Tuned prediction threshold to 0.4 to improve recall for diabetic patients from 65% to 73%
## Results
| Model | Accuracy | Diabetic Recall |
|---|---|---|
| Logistic Regression | 75% | 73% |
| Random Forest | 84% | 89% |

Random Forest selected as primary model.
## Why Recall matters more than Accuracy
If a diabetic patient is incorrectly predicted as non-diabetic, they go untreated which can lead to serious complications like kidney failure and blindness. Missing a sick person is more dangerous than a false alarm — that's why recall matters more than accuracy in healthcare.
## Live App
[Click here to use the app](https://diabetes-risk-predictor-kappxoenkprrurmfy73snq4.streamlit.app/)
## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit
