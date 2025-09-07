# Sustainable Groundwater Monitoring and Rainfall Prediction System

## Week 2 - ML Model Implementation (Linear Regression & Random Forest)

This project implements machine learning models to predict groundwater levels using historical data from the State Groundwater Boards. Both **Linear Regression** and **Random Forest Regressor** models are used for comparison.

---

## Dataset

The dataset contains groundwater monitoring data with the following features:

- `date` – Date of observation
- `latitude` – Latitude of the observation location
- `longitude` – Longitude of the observation location
- `state_code` – Code of the state
- `currentlevel` – Current groundwater level (target variable)

**Dataset path used:**  
`C:\Users\moort\Downloads\state-groundwater-boards-changes-in-depth-to-water-level.csv`

---

## Data Preprocessing

1. Convert the `date` column to datetime format.  
2. Extract `year` and `month` from the `date` column as features.  
3. Handle missing values using forward fill (`ffill`).  

---

## Features & Target

- **Features (X):** `year`, `month`, `latitude`, `longitude`, `state_code`  
- **Target (y):** `currentlevel`

---

## Train-Test Split

- Split the dataset into training and testing sets with `80%` for training and `20%` for testing.  
- `random_state=42` is used for reproducibility.  

---

## Machine Learning Models

### 1. Linear Regression

- Trained on the training dataset.  
- Predictions made on the test dataset.  
- Evaluated using:
  - **Mean Squared Error (MSE)**
  - **R² Score**

### 2. Random Forest Regressor

- Trained with 100 trees (`n_estimators=100`).  
- Predictions made on the test dataset.  
- Evaluated using:
  - **Mean Squared Error (MSE)**
  - **R² Score**

---

## Evaluation Results

| Model                | Mean Squared Error (MSE) | R² Score |
|---------------------|-------------------------|----------|
| Linear Regression    | `<lin_mse>`             | `<lin_r2>` |
| Random Forest        | `<rf_mse>`              | `<rf_r2>` |

> Replace `<lin_mse>`, `<lin_r2>`, `<rf_mse>`, `<rf_r2>` with actual results after running the code.

---

## Visualization

- **Linear Regression:** Shows the correlation between actual and predicted groundwater levels (blue scatter plot).  
- **Random Forest:** Shows the correlation between actual and predicted groundwater levels (green scatter plot).  

Both plots are displayed side by side for easy comparison.

---

## Requirements

- Python 3.x  
- Libraries:
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

---

## Usage

1. Place the dataset in the specified path.  
2. Run the script:  
   ```bash
   python groundwater_prediction.py
