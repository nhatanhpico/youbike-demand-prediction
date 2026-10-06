# Member 2 – Regression experiments and Ridge comparison

## How to check

```bash
pip install -r requirements.txt
python member2/regression_review.py
```

The script runs the notebook code up to "5. Final test evaluation" (the test split is not used)
and prints PASS / FAIL for 56 checks: data split, the six NumPy functions, a from-scratch rebuild
of the learning-rate experiment, Ridge, the selection rules, the Member 2 cell, and every number in Q2.
Current result: 56 / 56 passed.

## Results

| Learning rate | Train MSE | Validation MSE |
|---:|---:|---:|
| 0.001 | 382.6274 | **328.5898** |
| 0.01  | 369.6333 | 328.8761 |
| 0.1   | 369.5585 | 328.6335 |

| Model | Train MSE | Train MAE | Validation MSE | Validation MAE |
|---|---:|---:|---:|---:|
| Linear regression (lr = 0.001) | 382.6274 | 11.3266 | **328.5898** | 10.5309 |
| Ridge (alpha = 1.0) | 369.5585 | 10.9219 | 328.6258 | 10.4697 |

- Selected learning rate: **0.001** (lowest validation MSE).
- Selected regression model: **Linear Regression** (validation MSE lower by 0.0360).
- lr = 0.1 reaches the closed-form least-squares solution; lr = 0.001 has not converged after 1,500 iterations.
- Ridge with alpha = 1.0 is almost the same as least squares (max coefficient difference 0.0055).
- No overfitting: validation MSE < training MSE for both models, R² 0.7917 → 0.7719,
  and 493 negative validation predictions point to underfitting.

## Changes in `Group_01_YouBike.ipynb`

- Added one markdown cell and one code cell ("Member 2: regression checks for Q2") under
  *Optional exploration*. It uses training and validation data only and does not change any model or selection.
- Filled in the Q2 answer and Member 2's contribution row.
- No other cell, text or output was changed. The whole notebook was also run from a fresh kernel
  with this cell included: 0 errors, and all other outputs (including the test results) were identical.
