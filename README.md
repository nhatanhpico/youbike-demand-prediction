# YouBike Rental Forecasting around NTU and Gongguan

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhatanhpico/youbike-demand-prediction/blob/main/Group_01_YouBike_output.ipynb)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end machine learning project predicting hourly YouBike rental demand and classifying high-demand periods across 8 key stations in the National Taiwan University (NTU) and Gongguan areas of Taipei.

---

## 🚀 Quick Start in Google Colab

Click the badge above or use this link to open the fully executed notebook directly in Google Colab:
- **[Open `Group_01_YouBike_output.ipynb` in Colab](https://colab.research.google.com/github/nhatanhpico/youbike-demand-prediction/blob/main/Group_01_YouBike_output.ipynb)**

---

## 📌 Project Overview

At the beginning of each hour, YouBike operations teams must forecast rental volume for the upcoming 60 minutes to inform redistribution logistics and staffing:
1. **Task 1: Hourly Rental Regression**: Predict continuous `target_rentals_next_hour` for $[t, t+1\text{ hour})$. Evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
2. **Task 2: High-Demand Classification**: Predict binary `target_high_demand`, defined as rentals strictly greater than the station's historical training 75th percentile. Evaluated across varying classification decision thresholds ($0.3$, $0.5$, $0.7$) using Accuracy, Precision, Recall, and F1 score.

---

## 📊 Dataset & Temporal Split Protocol

- **Coverage**: 8 stations (`S01`–`S08`), September 1 to November 30, 2025 (91 calendar days).
- **Total Records**: 17,472 station-hours (17,280 model-ready rows; the first 24 hours per station lack complete 24h lag history and are excluded from modeling).
- **Strict Chronological Evaluation**:
  - **Training Set**: September 1 – October 31, 2025 (11,520 model-ready rows)
  - **Validation Set**: November 1 – November 15, 2025 (2,880 rows)
  - **Test Set**: November 16 – November 30, 2025 (2,880 rows)
- **Features Used (13 total)**:
  - 6 Numeric: `hour`, `day_of_week`, `is_weekend`, `rentals_previous_hour`, `rentals_same_hour_previous_day`, `rentals_mean_previous_24h`.
  - 7 Categorical: One-hot encoded station identifiers (`S02`–`S08`, omitting reference `S01` to avoid multicollinearity).

---

## 🛠️ Implemented Core NumPy Functions

Implemented directly using pure NumPy without helper libraries:
1. `fit_standardizer(X_train)`: Computes training-only mean and population standard deviation (`ddof=0`), replacing zero scales with 1.0.
2. `apply_standardizer(X, mean, scale)`: Transforms feature matrices $(X - \mu) / \sigma$ using fitted training statistics.
3. `linear_predict(X, w, b)`: Vectorized linear hypothesis $Xw + b$.
4. `mse_loss(y, prediction)`: Mean squared error $\frac{1}{N} \sum (\hat{y} - y)^2$.
5. `linear_gradients(X, y, w, b)`: Analytical MSE gradients:
   $$\nabla_w L = \frac{2}{N} X^T (\hat{y} - y), \quad \nabla_b L = \frac{2}{N} \sum (\hat{y} - y)$$
6. `gradient_step(w, b, grad_w, grad_b, learning_rate)`: First-order parameter update.

---

## 📈 Key Experimental Results

### 1. Learning Rate Search & Convergence
- `lr = 0.001`: Train MSE = `382.63`, Validation MSE = **`328.59`** (**Selected Best**)
- `lr = 0.010`: Train MSE = `369.63`, Validation MSE = `328.88`
- `lr = 0.100`: Train MSE = `369.56`, Validation MSE = `328.63`

### 2. Task 1 Regression Model Comparison
| Model | Split | MSE | MAE |
| :--- | :--- | :---: | :---: |
| **Linear Regression (NumPy GD)** | Train | 382.63 | 11.33 |
| **Linear Regression (NumPy GD)** | Validation | **328.59** | 10.53 |
| Ridge Regression ($\alpha=1.0$) | Train | 369.56 | 10.92 |
| Ridge Regression ($\alpha=1.0$) | Validation | 328.63 | 10.47 |

*Linear Regression achieved lowest validation MSE and was selected for final test reporting.*

### 3. Final Test Set Performance
- **Test MSE**: `293.30`
- **Test MAE**: `10.48`

---

## 💻 Local Environment Setup

If running locally instead of Google Colab:

```bash
# Clone the repository
git clone https://github.com/nhatanhpico/youbike-demand-prediction.git
cd youbike-demand-prediction

# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
pip install jupyterlab

# Launch JupyterLab
jupyter lab
```

---

## 📁 Repository Structure

```text
├── README.md                      # Project documentation and Colab badge
├── README_EN.md                   # Course assignment specification
├── Assignment_Guide_EN.md         # Detailed assignment grading guide
├── Group_01_YouBike_output.ipynb  # Executed notebook with outputs and answers
├── Group_01_YouBike.ipynb         # Clean completed submission notebook
├── YouBike_Starter_EN.ipynb       # Course starter notebook
├── youbike_hourly_english.csv     # Hourly YouBike rental dataset
├── requirements.txt               # Pinned package versions
├── data_dictionary.csv            # Detailed feature definitions
├── station_mapping.csv            # Station codes and names
├── station_summary.csv            # Station descriptive statistics
└── source_audit.csv               # Data provenance checksums
```

---

## 📜 Attribution & License

- **Source Data**: Taipei City Department of Transportation, [Taipei YouBike 2.0 Rental Records](https://data.gov.tw/dataset/150635), licensed under Taiwan Government Data Open License 1.0.
- Hourly aggregation and English station labels are educational adaptations for National Taiwan University (NTU).
