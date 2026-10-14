# Email Spam Classification

## Project Overview
This project compares traditional Machine Learning models and Deep Learning models to classify email messages as Spam or Ham.

## Models Used
1. Logistic Regression
2. Multinomial Naive Bayes
3. LSTM
4. DistilBERT

## Technologies
- Python
- Pandas
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Seaborn

## Dataset
The project uses an email spam classification dataset containing email messages and their categories.

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 96.71% |
| Multinomial Naive Bayes | 95.83% |
| LSTM | 97.76% |
| DistilBERT | 98.74% |

DistilBERT achieved the highest reported accuracy in our experiments.

## How to Run

Install the required libraries:

pip install pandas scikit-learn matplotlib seaborn torch transformers

Run the traditional models:

python traditional_ml.py

Run the LSTM model:

python lstm_model.py

Run interactive spam prediction:

python predict_spam.py

Run DistilBERT:

python distilbert_model.py
