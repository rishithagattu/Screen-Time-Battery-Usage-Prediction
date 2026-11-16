# Screen Time vs Battery Usage Prediction

A simple machine learning project that uses **Linear Regression** to study the relationship between screen time and battery usage.

## Dataset

`data/battery_data.csv` contains two columns:

- `screen_time`: screen usage duration in minutes
- `battery_used`: battery consumed in percentage points

The CSV contains a new illustrative sample dataset created for this project. It is not measured from a real phone.

## Project files

```text
screen_time_battery_linear_regression/
├── data/
│   └── battery_data.csv
├── battery_prediction.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run on Windows

Open a terminal in this project folder.

```cmd
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py battery_prediction.py
```

The script loads the CSV automatically, splits it into training and testing sets, trains a Linear Regression model, reports Mean Squared Error and R², prints actual vs predicted test values, saves predictions to `data/battery_predictions.csv`, and displays a plot.

## Model

- Input feature (X): `screen_time`
- Target (y): `battery_used`
- Algorithm: `LinearRegression`
- Test split: 20%
- Evaluation: Mean Squared Error (MSE) and R²

## Note

This is an educational demonstration using illustrative data. Actual battery consumption varies with device, apps, brightness, network conditions, battery health, and other factors. Do not interpret these predictions as real-device measurements.
