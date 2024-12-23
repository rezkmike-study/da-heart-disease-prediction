# ML Project - Heart Disease Prediction

This project aims to predict the risk of heart disease using a pre-trained Random Forest model. A Streamlit application is deployed for easy interaction, allowing users to input specific health parameters and get predictions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Heart disease prediction is essential for early detection and prevention. This project uses a Random Forest model trained on health data to assess the risk of heart disease.

## Features

The Streamlit app uses the following inputs for prediction:
- **Sex** (Male/Female)
- **Chest Pain Type** (Typical Angina, Atypical Angina, etc.)
- **Max Heart Rate**
- **Exercise Induced Angina** (Yes/No)
- **Oldpeak** (ST depression induced by exercise)
- **ST Slope** (Upsloping, Flat, Downsloping)

## Model Deployment

The best-performing Random Forest model (`best_random_forest_model.pkl`) is deployed as a Streamlit app, allowing users to input their health data and receive predictions.

## Usage

### Streamlit Application

The Streamlit app allows users to:
- Input specific health parameters.
- Get predictions on heart disease risk.

## Requirements

- Python 3.7 or higher
- pandas
- scikit-learn
- joblib
- streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Project

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the app at the URL provided in the terminal (e.g., http://localhost:8501).

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
