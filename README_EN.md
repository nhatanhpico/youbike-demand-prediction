# NTU and Gongguan YouBike Assignment Package (Version 3)

An English group assignment for hourly rental forecasting and high-volume classification. Maximum four students per group, one notebook submission, and one shared grade. This version adds a starter notebook and reduces required model comparisons and written questions. The dataset is unchanged.

## Getting started

1. Extract this package.
2. Read `Assignment_Guide_EN.md`.
3. Open `YouBike_Starter_EN.ipynb` beside the CSV, or use Colab and upload the CSV when prompted.
4. Complete the marked TODOs. The unfinished notebook intentionally raises `NotImplementedError`.
5. Run all cells in order and submit one notebook with outputs and four short English answers per group.

See the recommended tested runtime and setup below. A CPU is sufficient.

## Files

- **YouBike_Starter_EN.ipynb**: guided notebook with preprocessing and linear-regression TODOs, a complete logistic-regression example, model comparisons, plots, and submission prompts.

- **youbike_hourly_english.csv**: the unified hourly table. Start here.
- **Assignment_Guide_EN.md**: English scenario, tasks, evaluation protocol, suggested grading, and discussion questions.
- **data_dictionary.csv**: types, roles, and definitions for all columns.
- **station_mapping.csv**: English teaching names and original Chinese source names. Chinese names are retained only for provenance.
- **station_summary.csv**: monthly rental counts, activity coverage, assumed-zero counts, and training thresholds.
- **source_audit.csv**: record checks and SHA-256 hashes of the downloaded archives.
- **daily_city_counts.csv**: daily citywide rental totals for quality inspection.
- **sources.csv**: official monthly archive URLs and license attribution.
- **build_dataset.py**: reproducible aggregation script (Python and NumPy).
- **build_summary.json**: machine-readable build summary.

## Verified scope

| Item | Result |
|---|---:|
| Stations | 8 |
| Calendar days | 91 |
| Period | September 1–November 30, 2025 |
| Rows | 17,472 |
| Model-ready rows | 17,280 |
| Selected source rentals | 505,628 |
| Station-hours assigned an assumed zero | 2,765 |
| City-hours without any recorded borrowing | 0 |
| Selected station-days without any recorded borrowing | 0 |

| Split | Dates | All rows | Model-ready rows |
|---|---|---:|---:|
| Training | September 1–October 31 | 11,712 | 11,520 |
| Validation | November 1–15 | 2,880 | 2,880 |
| Test | November 16–30 | 2,880 | 2,880 |

The first 24 hours per station lack the full lag history. Filter `model_ready == 1` before modeling. Missing numeric values are empty CSV cells. All other hourly bins are retained. Counts include conventional and electric bikes.

## Important interpretation

At time t, the target counts rentals in [t, t+1 hour). Lag features use only earlier hourly bins. High demand means a count strictly above that station's training-period 75th percentile. The supplied label threshold is not a probability threshold.

No recorded station-hour transactions are treated as zero under an explicit teaching assumption, flagged in `target_zero_assumed`. Observed citywide activity and daily station activity do not establish complete reporting or continuous station operation. This package has not verified closure histories or weather explanations for unusual days. It supports forecasting observed rental volume, not identifying unmet demand or actual stockouts.

Do not use target, threshold, split, readiness, or audit fields as features. Read the allowed-input list in the assignment guide.

The three source files contain 20,831,273 rows in total. Two rows fail the borrowing-field/structure checks and are excluded. Rows with a valid borrowing timestamp and station are counted even when other fields are blank. The build preserves repeated-looking trip records because the source has no unique trip ID and timestamps have hourly precision. Aggregated totals reconcile to 505,628 selected source rentals.

## Rebuild

Download the three monthly archives listed in `sources.csv`. Save them as `youbike-202509.zip`, `youbike-202510.zip`, and `youbike-202511.zip` in one directory. Then run:

```bash
python build_dataset.py --raw-dir /path/to/archives --output-dir ./rebuilt
```

The raw citywide archives are not included. The script rebuilds the table and numeric audit files. Documentation and `sources.csv` are supplied separately in this package.

## Attribution

Source: Taipei City Department of Transportation, [Taipei YouBike 2.0 Rental Records](https://data.gov.tw/dataset/150635), Taiwan Government Data Open License 1.0. Hourly aggregation and English station labels are teaching adaptations. Sources accessed September 14, 2026.

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

