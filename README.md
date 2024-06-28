# Fake News Predictor Using Logistic Regression

This project aims to build a machine learning model to classify news articles as real or fake using Logistic Regression. By leveraging natural language processing (NLP) techniques and Logistic Regression, the project provides an effective tool for identifying potentially misleading information.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

The Fake News Predictor uses Logistic Regression to analyze the text of news articles and predict whether they are real or fake. This tool is designed to assist journalists, educators, researchers, developers, and the general public in identifying the reliability of news sources and articles.

## Features

- **Data Preprocessing**:
  - Text cleaning (removal of punctuation, stop words, etc.)
  - Tokenization
  - Stemming and Lemmatization
  - Vectorization (TF-IDF)
- **Model Training**:
  - Splitting the dataset into training and testing sets
  - Training a Logistic Regression model
  - Evaluating model performance using various metrics (accuracy, precision, recall, F1-score)
- **Prediction System**:
  - Input news articles for classification
  - Predicting the likelihood of the article being fake or real

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/fake-news-predictor.git
    cd fake-news-predictor
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the Fake News Predictor, follow these steps:

1. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook Fake\ News\ Prediction.ipynb
    ```

2. **Import relevant dependencies**:
    ```python
    import numpy as np
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    ```

3. **Train the model**:
    ```python
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    ```

4. **Evaluate model performance**:
    ```python
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(train_pred, Y_train)
    print(f'Train Accuracy: {train_acc * 100:.2f}%')

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(test_pred, Y_test)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')
    ```

5. **Make predictions**:
    ```python
    X_new = X_test[1234]
    pred = model.predict(X_new)

    if pred[0] == 0:
        print("The news is Real")
    else:
        print("The news is Fake")
    ```

