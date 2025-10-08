# 📧 Email Spam Classification

## Project Overview
This project builds a **spam detection system** that classifies emails as **Spam (1)** or **Not Spam (0)** using Python and machine learning. The workflow covers **data preprocessing, feature extraction, model training, evaluation, and prediction**, providing an end-to-end pipeline.

## Dataset
- **Source:** [Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)  
- **Content:** Email texts and labels (`0` = Not Spam, `1` = Spam)  
- **Purpose:** Train and evaluate machine learning models for spam detection

## Aim
To classify emails as spam or not by:
- Cleaning and preprocessing raw text  
- Converting text into numerical features  
- Training machine learning models  
- Evaluating and analyzing model performance

## Workflow
1. **Data Loading & Exploration** – Load dataset, check label distribution and missing values.  
2. **Data Preprocessing** – Lowercase text, remove punctuation, numbers, URLs, and stopwords.  
3. **Feature Extraction** – Transform text into numerical vectors using TF-IDF.  
4. **Train/Test Split** – Split data into training and testing sets while maintaining label distribution.  
5. **Model Training** – Train Multinomial Naive Bayes and Logistic Regression models.  
6. **Model Evaluation** – Evaluate using Accuracy, ROC-AUC, Precision, Recall, and F1-score.  
7. **Error Analysis** – Inspect misclassified emails to improve the model.  
8. **Prediction** – Classify new emails in real-time using the trained model.

## Results
| Model               | Accuracy | ROC-AUC  | Precision (Spam) | Recall (Spam) | F1 (Spam) |
| ------------------- | -------- | -------- | ---------------- | ------------- | --------- |
| Naive Bayes         | 96.66%   | 0.993    | 96.67%           | 96.98%        | 96.83%    |
| Logistic Regression | 98.23%   | 0.997    | 97.71%           | 98.95%        | 98.33%    |

**Observation:** Logistic Regression achieved the best overall performance, minimizing false negatives.

## Future Enhancements
- Experiment with deep learning models (LSTM, Transformers)  
- Use ensemble methods for improved accuracy  
- Deploy as a web application or API for email filtering  

## Tech Stack
- **Language:** Python  
- **Libraries:** scikit-learn, pandas, numpy, nltk  
- **Models:** Multinomial Naive Bayes, Logistic Regression  

## Conclusion
The project demonstrates a complete **machine learning pipeline for email spam detection**, achieving over **98% accuracy** with Logistic Regression. It can be extended for real-world email filtering systems.

## Dataset
[Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)  

**Author:** Dhriti Gupta
