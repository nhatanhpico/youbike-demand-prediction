# YouBike Rental Forecasting around NTU and Gongguan

## Scenario

You are helping a YouBike operations team identify busy stations around National Taiwan University (NTU) and Gongguan. At the beginning of each hour, predict how many rentals will start during the next 60 minutes and whether this will be a high-rental-volume period for that station. Your predictions can inform staffing and redistribution planning.

This assignment connects linear regression, logistic regression, gradient descent, feature representation, regularization, decision trees, and random forests. Grades depend on correct implementation, fair evaluation, and evidence-based explanations. There is no leaderboard or performance threshold to beat. A well-designed experiment can receive full credit even if a more complex model does not improve performance.

## Dataset and prediction time

The dataset contains eight stations from September 1 to November 30, 2025. Each row represents one station and one forecast hour. All timestamps use Asia/Taipei time (UTC+08:00).

For example, a row with `forecast_time = 2025-10-15T09:00:00+08:00` predicts rentals starting in **[09:00, 10:00)**. `rentals_previous_hour` counts rentals starting in [08:00, 09:00). The previous-day feature counts rentals in [09:00, 10:00) on October 14. The rolling mean uses the 24 completed hourly bins immediately before the forecast time.

Both conventional and electric bikes are included. Each source record counts as one rental starting at its borrowing station. English station names are teaching translations. `station_mapping.csv` preserves the original Chinese station names for traceability. The S01–S08 identifiers are teaching identifiers, not official YouBike station codes.

### Task 1: Regression

Predict `target_rentals_next_hour`, the number of observed rentals starting during the forecast hour. Report MSE and MAE. Keep predictions continuous when computing these metrics. For consistent comparisons, use raw predictions without rounding or clipping, and discuss negative predictions if they occur.

### Task 2: Classification

Predict `target_high_demand`. For each station, the dataset defines a threshold as the 75th percentile of its hourly rental counts in the training period, using NumPy's linear percentile interpolation. A target count **strictly greater than** this threshold receives label 1. A count equal to or below the threshold receives label 0. The same station-specific threshold applies to all splits. Because counts are discrete, the positive class need not represent exactly 25% of rows.

This count threshold defines the ground-truth label. It is different from the probability threshold, such as 0.5, used to convert a classifier's predicted probability into a class decision.

## Fixed evaluation protocol

| Split | Forecast dates (inclusive) | Purpose |
|---|---|---|
| Training | September 1–October 31, 2025 | Fit model parameters and preprocessing |
| Validation | November 1–15, 2025 | Choose model settings and probability threshold |
| Test | November 16–30, 2025 | Evaluate after all choices are fixed |

Use rows with `model_ready = 1`. The first 24 hours for each station do not have complete lag features. Keep these rows in the supplied table for traceability, but exclude them from modeling. Do not shuffle the full dataset to create a new split. Shuffling training rows for optimization is allowed.

This is rolling one-hour-ahead evaluation. Each new forecast may use actual rental counts from earlier completed hours, including earlier hours of the validation or test period. Do not use any count from the forecast hour or a later hour as an input. This protocol does not represent predicting an entire month in advance.

The exercise assumes that counts for completed hours are available when each forecast is made. The monthly archive itself is retrospective and does not establish real-time reporting latency.

Fit standardization on training data only. Apply those same parameters to validation and test data. If you scale the regression target, fit that scaling on training data and convert predictions back to rental counts before evaluation. Never standardize the binary label. Fit polynomial transformations and scaling within the training workflow.

### Allowed model inputs

- `station_id`, encoded with one-hot encoding. Use a fixed station vocabulary and omit one reference category when including an intercept. Do not treat S01–S08 as a numerical order.
- `hour`, `day_of_week`, and `is_weekend`.
- `rentals_previous_hour`, `rentals_same_hour_previous_day`, and `rentals_mean_previous_24h`.

Use `forecast_time` for sorting and time alignment. `station_name` is a readable identifier and duplicates the information in `station_id`. Do not use targets, `high_demand_threshold`, `split`, `model_ready`, or quality flags as model inputs. The quality flags use retrospective information and are for auditing only.

## Environment and reproducibility

Recommended tested runtime: **Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, matplotlib 3.10.8, scikit-learn 1.8.0**. No GPU is required.

Recommended local setup (after installing Python 3.12.14):

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell (instead of the line above):
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab
```

`requirements.txt` fixes the four scientific-package versions; it does not install Python or lock all transitive dependencies. JupyterLab is a notebook interface and is installed separately. Another environment, including Colab, is allowed. Do not change a working Colab runtime merely to match package versions. **Every group must execute the final environment cell and retain its Python version, key package versions, and complete `pip list` output.** No separate environment file is required. Small numerical differences alone are not penalized.

## Submission

Submit **one executed notebook named `Group_01_YouBike.ipynb`**, replacing `01` with your group number. Preserve all code, figures, tables, four short English answers, contribution statements, and environment output. Do not submit the CSV, ZIP, or an exported Python script. Use the supplied CSV unchanged. Keep function names, signatures, and the provided experiment sections so TAs can rerun your work. Implement the six required functions using NumPy directly, without additional custom helper functions. Do not remove reference cells or the environment cell.

The basic self-checks are formative, not the complete marking tests. TAs check core functions using independent numerical tests and review the remaining experiments and explanations. Submitted output text alone is not evidence of correctness. A run failure is flagged for review and does not automatically assign a zero to the entire assignment.

## Group work and submission

Use your existing course group, with a maximum of four students. Submit one executed notebook per group. All members receive the same assignment grade. Every member should understand both prediction tasks and the main results. Individual understanding may be assessed in the regular course exams. There is no separate individual check for this assignment.

At the end of the notebook, list each member's name, student ID, and one or two sentences describing their specific contribution. Include everyone in implementation, experimentation, or interpretation rather than assigning a member only document formatting.

Download and extract the complete package. Open `YouBike_Starter_EN.ipynb` with `youbike_hourly_english.csv` in the same folder. For Google Colab, upload the CSV when prompted. NumPy, pandas, matplotlib, and scikit-learn are required. No GPU is required.

The starter notebook provides loading, station encoding, the training loop, logistic regression, metrics, plots, and comparison tables. Cells marked **TODO** identify required student work. The notebook intentionally stops at unfinished functions until you implement them. It is not a completed solution.

## Required tasks and grading

| Part | Required work | Weight |
|---|---|---:|
| 1. Data preparation | Follow the supplied time split and complete training-only standardization | 15% |
| 2. Linear regression | Complete prediction, MSE, gradients, and parameter updates. Run the three supplied learning-rate experiments and interpret the loss curves | 30% |
| 3. Classification | Run the supplied logistic regression implementation. Compare probability thresholds 0.3, 0.5, and 0.7 using validation data | 20% |
| 4. Model comparison | Compare linear regression with Ridge, and logistic regression with a decision tree. Complete the result tables and select models using validation results | 20% |
| 5. Interpretation and contribution statement | Answer the four questions below and provide each member's contribution | 15% |

The linear regression training loop is provided. Implement the core computations in NumPy. Use MSE averaged across samples, with gradients matching that normalization. Scikit-learn is allowed for Ridge, the decision tree, and evaluation. The logistic regression implementation is provided for you to run, examine, and interpret.

Keep the supplied experiment settings: three linear-regression learning rates, Ridge alpha = 1.0, and a decision tree with maximum depth 5. Select the linear-regression learning rate using validation MSE. Select between linear regression and Ridge using validation MSE. Compare classification methods at probability threshold 0.5, then select the final method and threshold from the supplied validation threshold table using F1. Use the notebook's deterministic tie-breaking order.

After all choices are fixed, run the final test section once. Test results are used for reporting and discussion, not further tuning. Report MSE and MAE for regression. For classification, report accuracy, precision, recall, F1, and a confusion matrix. The starter notebook supplies these calculations. No improvement over another model is required for full credit.

## Four discussion questions

Write approximately 3–5 English sentences per question. Refer to your own figures or numerical results. Clear technical reasoning matters more than grammar.

1. **Data and leakage:** Explain why the split is chronological and why standardization uses training statistics only. Identify one supplied column that would leak target information if used as a feature.
2. **Learning and generalization:** Describe the effect of learning rate in your loss curves. Compare training and validation results for linear regression and Ridge. State whether the results support an overfitting diagnosis, and explain why.
3. **Classification decisions:** Explain how precision and recall change as the logistic regression probability threshold changes. Do its learned parameters change? Compare logistic regression with the decision tree at threshold 0.5 and justify your final validation-based choice.
4. **Failure analysis:** Select one test case that your final model predicts poorly. Report its station, forecast time, actual target, and prediction. Give a possible explanation, distinguish it from verified evidence, and explain why rental volume alone cannot establish that a station had no bikes available.

## Optional exploration

Polynomial features and random forests are optional and are not required for full credit. You may also compare classification based on a regression prediction against direct logistic classification. Complete the required work first. Any optional exploration must use validation data for model choices and preserve the test protocol. If you explore these extensions, run them before the final test section.

## Submission checklist

- One `Group_XX_YouBike.ipynb` file per group, with outputs, environment information, and English answers preserved. Do not submit the CSV.
- All required TODOs completed, with cells executed in order from a fresh session.
- Required figures, validation tables, final model choices, and test results visible.
- Four short answers and a contribution statement for every member.
- No private answer keys or external solutions included.

## Data interpretation and limitations

The target measures completed observed rentals recorded in the archive. It does not measure unmet demand: a person who could not find a bike generates no rental record. Bike availability also depends on returns, initial inventory, and operator redistribution. This assignment predicts high rental volume, not stockouts or an optimal redistribution policy.

An hourly station bin with no recorded rentals is assigned zero only when the archive contains at least one rental somewhere in the city during that hour. Such rows have `target_zero_assumed = 1`. This is a teaching assumption, not proof that the station was open or that reporting was complete. Citywide activity and station activity on the same day do not rule out partial reporting gaps or temporary closures. If an entire city hour has no records, the count is left blank and the affected forecast/lag rows are not model-ready.

The package retains low-volume days and includes daily city totals for inspection. Do not remove a low-volume day simply because its count is unusual. Weather, closures, and reporting issues require separate evidence. Weather and holiday indicators are not included in this first version. Weekend is determined from the day of the week and does not identify public holidays.

The source has hour-level timestamps and no unique trip identifier in the supplied seven fields. Identical-looking records may represent separate trips, so the build does not automatically remove duplicate-looking rows. It uses valid borrowing time and station fields even if unrelated fields are blank. Source-level completeness is not independently verified.

## Sources

- Taipei City Department of Transportation, [Taipei YouBike 2.0 Rental Records](https://data.gov.tw/dataset/150635).
- [Taipei Open Data dataset page](https://data.taipei/dataset/detail?id=c5924c17-25db-4f1e-99c4-f8ada40f2445).
- Monthly download URLs and SHA-256 archive checksums are supplied in `sources.csv` and `source_audit.csv`. The source dataset specifies Taiwan Government Data Open License, version 1.0. The English names and hourly aggregation are teaching adaptations.
