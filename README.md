
# Fake News Detection Using NLP

## Project Overview
Fake News Detection Using NLP is a machine learning-based application that identifies whether a given news article is Fake or Real.

The system uses Natural Language Processing (NLP) techniques and Machine Learning algorithms to analyze news content and predict its authenticity.

## Objectives
- To identify fake and real news articles.
- To apply NLP techniques for text processing.
- To train a machine learning model.
- To provide predictions through a user-friendly web application.

## Technologies Used
- Python
- Natural Language Processing (NLP)
- Machine Learning
- Scikit-learn
- Pandas
- Streamlit
- TF-IDF Vectorization
- Logistic Regression

## Dataset
The project uses two datasets:
- Fake.csv – Fake news articles
- True.csv – Real news articles

## Algorithm
Logistic Regression

## Methodology
1. Collect fake and real news datasets.
2. Preprocess the news text.
3. Combine title and article content.
4. Convert text into numerical features using TF-IDF.
5. Train the Logistic Regression model.
6. Evaluate model performance.
7. Predict whether news is Fake or Real.

## Project Structure

Fake-News-Detection-Using-NLP/
- app.py
- train_model.py
- requirements.txt
- README.md
- dataset/
- model/

## How to Run

Install the required libraries:

pip install -r requirements.txt

Train the model:

python train_model.py

Run the application:

streamlit run app.py

## Output
The application predicts:
- Fake News
- Real News

It also displays the model confidence score.

## Developed By
AI & Data Science Student

## Conclusion
This project demonstrates how Natural Language Processing and Machine Learning can be used to identify potentially fake news articles.
