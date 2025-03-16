Automated Data Analysis & Machine Learning App

📌 Overview

This Streamlit-based web application automates data analysis and machine learning model training. Users can upload a dataset, clean the data, perform exploratory data analysis (EDA), and train machine learning models—all within an intuitive UI.

📂 Project Structure

├── app.py              # Main Streamlit app
├── data_loader.py      # Handles data loading
├── data_cleaning.py    # Cleans and preprocesses data
├── eda.py              # Performs exploratory data analysis
├── ml_model.py         # Trains and evaluates machine learning models

🚀 Features

Upload CSV files to analyze.

Data Cleaning: Handle missing values, remove duplicates, outlier detection, type conversion, text cleaning, and feature scaling.

EDA (Exploratory Data Analysis): View dataset overview, correlation matrix, histograms, scatter plots, and box plots.

Machine Learning: Supports classification and regression models with hyperparameter tuning.

Interactive UI: Uses Streamlit for easy navigation and real-time feedback.


📥 Installation & Usage

1️⃣ Install Dependencies

Ensure you have Python installed, then run:

pip install streamlit pandas numpy seaborn matplotlib plotly scikit-learn

2️⃣ Run the App

streamlit run app.py

3️⃣ Upload Your Dataset

Click the "Upload your dataset (CSV)" button.

The app will load, clean, and analyze your data automatically.


4️⃣ Explore and Train Models

Use sidebar options to clean data and perform EDA.

Choose a target variable and machine learning model.

Train the model and view evaluation metrics.


🤖 Supported Machine Learning Models

Classification: Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes

Regression: Linear Regression, Decision Tree, Random Forest, SVM


📌 Notes

The app automatically detects whether the task is classification or regression based on the target variable.

Tree-based models display feature importance after training.


🛠 Future Improvements

Add support for deep learning models.

Implement automatic feature engineering.

Provide downloadable model files after training.




