# Spam Email Detection using Machine Learning

This project implements a spam detection system that classifies SMS messages as **Spam** or **Not Spam (Ham)** using machine learning techniques.

## Project Overview

The model is trained on the SMS Spam Collection dataset.  
Text messages are converted into numerical features using TF-IDF vectorization, and a Support Vector Machine (SVM) classifier is used for prediction.

## Features

- Text preprocessing using regular expressions
- TF-IDF feature extraction
- Support Vector Machine (SVM) classifier
- Model evaluation using classification report and confusion matrix
- Interactive user input system for real-time prediction

## Model Performance

The model achieves approximately **99% accuracy** on the test dataset.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## How to Run the Project

1. Install required libraries:


pip install -r requirements.txt
