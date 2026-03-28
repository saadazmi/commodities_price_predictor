Commodities Price Predictor

This project is a Machine Learning–based web application that predicts the prices of used cars and new laptops based on their specifications.

Users can enter product details such as brand, specifications, and other attributes, and the system will generate an estimated price prediction using trained machine learning models.

The project integrates data preprocessing, model training, and a Flask-based web interface to provide an interactive prediction system.

Technologies Used

Python – Used for data processing, machine learning, and backend development.

Pandas – Used for data cleaning, manipulation, and preprocessing of datasets.

NumPy – Used for numerical operations and array handling.

Scikit-learn – Used to build and train machine learning models for price prediction.

Flask – A lightweight web framework used to connect the machine learning model with the web interface.

HTML – Used to design the structure of the web pages.

CSS – Used for styling and improving the visual appearance of the web application.

Dataset

The datasets used in this project are:

quikr_car.csv

laptop_data.csv

These datasets were cleaned and preprocessed to remove unnecessary data, handle missing values, and prepare the features for model training.

After preprocessing, the cleaned datasets were saved as:

Cleaned_Car_data.csv

Cleaned_laptop_data.csv

These cleaned datasets were used to train the machine learning models.

Machine Learning Models

The trained models are stored as serialized .pkl files:

car_model.pkl – Model trained to predict car prices.

laptop_model.pkl – Model trained to predict laptop prices.

These models are loaded in the Flask application to generate predictions based on user inputs.

Project Structure Commodities_Price_Predictor │ ├── app.py ├── car_model.pkl ├── laptop_model.pkl │ ├── quikr_car.csv ├── laptop_data.csv ├── Cleaned_Car_data.csv ├── Cleaned_laptop_data.csv │ ├── templates │ └── index.html │ ├── static │ └── style.css

app.py – Main Flask application that connects the frontend with the machine learning models.

templates/ – Contains HTML files used to build the user interface.

static/ – Contains CSS and other static files used for styling.

Cleaned datasets – Used for training and reference.

How to Run the Project

Install required libraries pip install pandas numpy scikit-learn flask
Run the Flask application python app.py
Open the application in your browser http://127.0.0.1:5000/
