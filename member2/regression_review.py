# Member 2: verification of the regression part of Group_01_YouBike.ipynb
# Run from the repo root:  python member2/regression_review.py
# The notebook cells are executed up to (not including) "5. Final test evaluation",
# so the test split is never used here.

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")                 # no plot window
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out_dir = Path(__file__).resolve().parent
nb_path = root / "Group_13_YouBike.ipynb"
results = {}
checks = {}                                                # check name -> True / False


def check(name, ok):
    checks[name] = bool(ok)
    print(("PASS  " if ok else "FAIL  ") + name)


# ---------- Load the notebook code ----------
cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
code_cells = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
stop = next(i for i, src in enumerate(code_cells) if 'regression_prediction(selected_reg, "test")' in src)
m2_index = next(i for i, src in enumerate(code_cells) if "Member 2: extra checks" in src)
answers = next("".join(c["source"]) for c in cells if "**Q2. Learning and generalization.**" in "".join(c["source"]))

nb = {}                                                    # notebook variables
os.chdir(root)
for i in range(stop):
    if i == m2_index:                                      # save the state before the Member 2 cell runs
        before = {"best_lr": nb["best_lr"], "selected_reg": nb["selected_reg"],
                  "selected_cls": nb["selected_cls"], "selected_threshold": nb["selected_threshold"],
                  "w": nb["linear"]["w"].copy(), "b": nb["linear"]["b"], "ridge": nb["ridge"].coef_.copy()}
    exec(compile(code_cells[i], f"<cell {i}>", "exec"), nb)
    nb["display"] = lambda *a, **k: None                   # keep the output short
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.close("all")

X, X_raw, y, parts = nb["X"], nb["X_raw"], nb["y_reg"], nb["parts"]
fit_standardizer, apply_standardizer = nb["fit_standardizer"], nb["apply_standardizer"]
linear_predict, mse_loss = nb["linear_predict"], nb["mse_loss"]
linear_gradients, gradient_step = nb["linear_gradients"], nb["gradient_step"]
learning_rates = nb["learning_rates"]

# ---------- 1. Data split and features ----------
print("\n1. Data split and features")
raw = pd.read_csv(root / "youbike_hourly_english.csv")
raw["forecast_time"] = pd.to_datetime(raw["forecast_time"])
dates = {"train": ("2025-09-01", "2025-10-31"), "validation": ("2025-11-01", "2025-11-15"),
         "test": ("2025-11-16", "2025-11-30")}
for split, (start, end) in dates.items():
    day = parts[split]["forecast_time"].dt.strftime("%Y-%m-%d")
    check(f"{split} dates inside {start} .. {end}", day.min() >= start and day.max() <= end)
    check(f"{split} rows all model_ready", parts[split]["model_ready"].eq(1).all())
    expected = raw[(raw["split"] == split) & (raw["model_ready"] == 1)]
    check(f"{split} row count = CSV ({len(expected)})", len(parts[split]) == len(expected))
check("splits do not overlap in time",
      parts["train"]["forecast_time"].max() < parts["validation"]["forecast_time"].min()
      and parts["validation"]["forecast_time"].max() < parts["test"]["forecast_time"].min())
station_cols = X_raw["train"][:, 6:]                       # 7 one-hot columns S02..S08
s01_rows = parts["train"]["station_id"].eq("S01").to_numpy()
check("one-hot: S01 rows are all zero", (station_cols[s01_rows] == 0).all())
check("one-hot: other rows have exactly one 1", (station_cols[~s01_rows].sum(axis=1) == 1).all())
check("first 6 columns are the allowed numeric features",
      np.array_equal(X_raw["train"][:, :6], parts["train"][nb["numeric_features"]].to_numpy(float)))

# ---------- 2. The six NumPy functions ----------
print("\n2. The six NumPy functions")
rng = np.random.default_rng(0)
ok_std, ok_pred, ok_mse, ok_grad, ok_step, ok_shape = True, True, True, True, True, True
for n, d in [(5, 1), (50, 4), (200, 13)]:                  # several shapes
    X_test = rng.normal(3, 2, size=(n, d))
    y_test = rng.normal(size=n)
    w_test, b_test = rng.normal(size=d), float(rng.normal())
    mean, scale = fit_standardizer(X_test)
    ok_std &= np.allclose(mean, X_test.sum(0) / n)
    ok_std &= np.allclose(scale, np.sqrt(((X_test - X_test.sum(0) / n) ** 2).sum(0) / n))   # ddof = 0
    ok_std &= np.allclose(apply_standardizer(X_test, mean, scale), (X_test - mean) / scale)
    y_hat = linear_predict(X_test, w_test, b_test)
    ok_pred &= np.allclose(y_hat, [sum(X_test[i, j] * w_test[j] for j in range(d)) + b_test for i in range(n)])
    ok_mse &= np.isclose(mse_loss(y_test, y_hat), mean_squared_error(y_test, y_hat))
    grad_w, grad_b = linear_gradients(X_test, y_test, w_test, b_test)
    ok_shape &= grad_w.shape == (d,) and np.ndim(grad_b) == 0
    eps = 1e-6                                             # central finite difference
    num_w = [(mse_loss(y_test, linear_predict(X_test, w_test + eps * e, b_test))
              - mse_loss(y_test, linear_predict(X_test, w_test - eps * e, b_test))) / (2 * eps) for e in np.eye(d)]
    num_b = (mse_loss(y_test, linear_predict(X_test, w_test, b_test + eps))
             - mse_loss(y_test, linear_predict(X_test, w_test, b_test - eps))) / (2 * eps)
    ok_grad &= np.allclose(grad_w, num_w, rtol=1e-5, atol=1e-6) and np.isclose(grad_b, num_b, rtol=1e-5, atol=1e-6)
    w_copy = w_test.copy()
    w_new, b_new = gradient_step(w_test, b_test, grad_w, grad_b, 0.05)
    ok_step &= np.allclose(w_new, w_copy - 0.05 * grad_w) and np.isclose(b_new, b_test - 0.05 * grad_b)
    ok_step &= np.array_equal(w_test, w_copy)               # input weights are not changed in place
check("fit_standardizer / apply_standardizer (mean, population std)", ok_std)
check("zero std is replaced by 1", fit_standardizer(np.array([[1.0, 7.0], [3.0, 7.0]]))[1][1] == 1.0)
check("linear_predict = X w + b", ok_pred)
check("mse_loss = sklearn mean_squared_error", ok_mse)
check("linear_gradients = finite differences", ok_grad)
check("gradient shapes: grad_w (D,), grad_b scalar", ok_shape)
check("gradient_step = w - lr * grad, inputs unchanged", ok_step)
mean_tr, scale_tr = X_raw["train"].mean(0), X_raw["train"].std(0)
check("all splits use training mean / std only",
      all(np.allclose(X[s], (X_raw[s] - mean_tr) / scale_tr) for s in ["train", "validation", "test"]))

# ---------- 3. Training loop rebuilt from scratch ----------
print("\n3. Learning-rate experiment rebuilt without the notebook functions")
lr_table = nb["lr_table"].set_index("learning_rate")
X_train_b = np.column_stack([X["train"], np.ones(len(X["train"]))])
X_val_b = np.column_stack([X["validation"], np.ones(len(X["validation"]))])
for lr in learning_rates:
    theta = np.zeros(X_train_b.shape[1])
    for step in range(1500):
        theta -= lr * 2 / len(X_train_b) * X_train_b.T @ (X_train_b @ theta - y["train"])
    train_mse = np.mean((X_train_b @ theta - y["train"]) ** 2)
    val_mse = np.mean((X_val_b @ theta - y["validation"]) ** 2)
    check(f"lr={lr}: train / validation MSE match the notebook table",
          np.isclose(train_mse, lr_table.loc[lr, "train_mse"], rtol=1e-10)
          and np.isclose(val_mse, lr_table.loc[lr, "validation_mse"], rtol=1e-10))
    if lr == nb["best_lr"]:
        check(f"lr={lr}: weights match the notebook model",
              np.allclose(theta[:-1], nb["linear"]["w"]) and np.isclose(theta[-1], nb["linear"]["b"]))
results["lr_table"] = nb["lr_table"].to_dict("records")

# ---------- 4. Ridge ----------
print("\n4. Ridge (alpha = 1.0)")
ridge = nb["ridge"]
X_c = X["train"] - X["train"].mean(0)                      # sklearn does not penalize the intercept
w_ridge = np.linalg.solve(X_c.T @ X_c + np.eye(X_c.shape[1]), X_c.T @ (y["train"] - y["train"].mean()))
check("sklearn Ridge = closed-form ridge", np.allclose(ridge.coef_, w_ridge, atol=1e-8))
check("Ridge uses alpha = 1.0", ridge.alpha == 1.0)
reg_table = nb["reg_table"]
for model in ["Linear regression", "Ridge"]:
    for split in ["train", "validation"]:
        y_pred = nb["regression_prediction"](model, split)
        row = reg_table[(reg_table["model"] == model) & (reg_table["split"] == split)].iloc[0]
        check(f"{model} {split}: MSE and MAE in reg_table",
              np.isclose(row["mse"], np.mean((y[split] - y_pred) ** 2))
              and np.isclose(row["mae"], np.mean(np.abs(y[split] - y_pred))))
results["reg_table"] = reg_table.to_dict("records")

# ---------- 5. Model selection ----------
print("\n5. Selection rules")
best_lr = min(learning_rates, key=lambda lr: (lr_table.loc[lr, "validation_mse"], lr))
check(f"best learning rate by validation MSE = {best_lr}", best_lr == nb["best_lr"])
val_rows = reg_table[reg_table["split"] == "validation"].set_index("model")["mse"]
best_model = "Linear regression" if val_rows["Linear regression"] <= val_rows["Ridge"] else "Ridge"   # tie -> linear
check(f"regression model by validation MSE = {best_model}", best_model == nb["selected_reg"])
results["selected_learning_rate"] = best_lr
results["selected_regression_model"] = best_model

# ---------- 6. Numbers in the Member 2 cell ----------
print("\n6. Member 2 cell")
theta_ls = np.linalg.lstsq(X_train_b, y["train"], rcond=None)[0]
gap_table = nb["gap_table"].set_index(["model", "split"])
for model in ["Linear regression", "Ridge"]:
    for split in ["train", "validation"]:
        y_pred = nb["regression_prediction"](model, split)
        mse = np.mean((y[split] - y_pred) ** 2)
        row = gap_table.loc[(model, split)]
        check(f"gap_table {model} {split}",
              np.isclose(row["mse"], mse) and np.isclose(row["target_variance"], y[split].var())
              and np.isclose(row["r2"], 1 - mse / np.mean((y[split] - y[split].mean()) ** 2))
              and row["negative_predictions"] == int((y_pred < 0).sum()))
check("least squares solution matches notebook (theta_ls)", np.allclose(theta_ls, nb["theta_ls"]))
check("lr = 0.1 reaches the least squares solution",
      np.isclose(lr_table.loc[0.1, "train_mse"], np.mean((X_train_b @ theta_ls - y["train"]) ** 2), rtol=1e-9))
eig_vals = np.linalg.eigvalsh(2 / len(X_train_b) * X_train_b.T @ X_train_b)
lr_limit = 2 / eig_vals.max()
check(f"all learning rates below the stable limit {lr_limit:.4f}", max(learning_rates) < lr_limit)
check("Member 2 cell did not change any model or selection",
      before["best_lr"] == nb["best_lr"] and before["selected_reg"] == nb["selected_reg"]
      and before["selected_cls"] == nb["selected_cls"] and before["selected_threshold"] == nb["selected_threshold"]
      and np.array_equal(before["w"], nb["linear"]["w"]) and before["b"] == nb["linear"]["b"]
      and np.array_equal(before["ridge"], nb["ridge"].coef_))

# ---------- 7. Every number written in the Q2 answer ----------
print("\n7. Numbers in the Q2 answer")
q2 = answers.split("**Q2. Learning and generalization.**")[1].split("**Q3.")[0]
lin_val, rdg_val = val_rows["Linear regression"], val_rows["Ridge"]
claims = {                                                 # text in Q2 -> value computed here
    "369.5585": lr_table.loc[0.1, "train_mse"],
    "382.6274": lr_table.loc[0.001, "train_mse"],
    "0.368": lr_limit,
    "328.5898": lr_table.loc[0.001, "validation_mse"],
    "328.8761": lr_table.loc[0.01, "validation_mse"],
    "328.6335": lr_table.loc[0.1, "validation_mse"],
    "0.04": lr_table.loc[0.1, "validation_mse"] - lr_table.loc[0.001, "validation_mse"],
    "328.6258": rdg_val,
    "0.0360": rdg_val - lin_val,
    "0.7917": gap_table.loc[("Linear regression", "train"), "r2"],
    "0.7719": gap_table.loc[("Linear regression", "validation"), "r2"],
    "493": gap_table.loc[("Linear regression", "validation"), "negative_predictions"],
}
for text, value in claims.items():
    places = len(text.split(".")[1]) if "." in text else 0
    check(f"Q2 says {text} (computed {value:.6f})", text in q2 and abs(float(text) - value) <= 0.5 * 10 ** -places)
check("Q2: Ridge training MSE also rounds to 369.5585", round(reg_table.iloc[2]["mse"], 4) == 369.5585)
check("Q2: validation MSE < training MSE for both models",
      all(gap_table.loc[(m, "validation"), "mse"] < gap_table.loc[(m, "train"), "mse"] for m in ["Linear regression", "Ridge"]))
check("Q2: lr = 0.001 has not converged (train MSE above least squares)",
      lr_table.loc[0.001, "train_mse"] > lr_table.loc[0.1, "train_mse"] + 1)
numbers_in_q2 = set(re.findall(r"\d[\d,]*\.?\d*", q2.split("**Answer:**")[1]))
results["q2_numbers_found"] = sorted(numbers_in_q2)

# ---------- Figure: train and validation MSE per iteration ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for lr in learning_rates:
    w, b = np.zeros(X["train"].shape[1]), 0.0
    train_curve, val_curve = [], []
    for step in range(1500):
        grad_w, grad_b = linear_gradients(X["train"], y["train"], w, b)
        w, b = gradient_step(w, b, grad_w, grad_b, lr)
        train_curve.append(mse_loss(y["train"], linear_predict(X["train"], w, b)))
        val_curve.append(mse_loss(y["validation"], linear_predict(X["validation"], w, b)))
    ax[0].plot(train_curve, label=f"lr={lr}")
    ax[1].plot(val_curve, label=f"lr={lr}")
ax[0].set_ylabel("Training MSE")
ax[1].set_ylabel("Validation MSE")
for a in ax:
    a.set_xlabel("Gradient descent iteration")
    a.set_ylim(320, 600)
    a.legend()
fig.tight_layout()
fig.savefig(out_dir / "lr_train_val_curves.png", dpi=130)

results["checks"] = checks
(out_dir / "regression_review_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
failed = [name for name, ok in checks.items() if not ok]
print(f"\n{len(checks) - len(failed)} / {len(checks)} checks passed")
raise SystemExit(1 if failed else 0)
