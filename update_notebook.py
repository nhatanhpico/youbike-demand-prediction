import json

with open("YouBike_Starter_EN.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 6: Standardization implementation
cell_6_code = """def fit_standardizer(X_train):
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale

def apply_standardizer(X, mean, scale):
    return (X - mean) / scale

feature_mean, feature_scale = fit_standardizer(X_raw["train"])
X = {s: apply_standardizer(a, feature_mean, feature_scale)
     for s, a in X_raw.items()}
assert np.isfinite(X["train"]).all()
print("Training shape:", X["train"].shape)
print("Maximum absolute training feature mean:", np.abs(X["train"].mean(axis=0)).max())
"""

# Cell 8: Linear regression implementation
cell_8_code = """def linear_predict(X, w, b):
    return X @ w + b

def mse_loss(y, prediction):
    return np.mean((prediction - y) ** 2)

def linear_gradients(X, y, w, b):
    prediction = linear_predict(X, w, b)
    error = prediction - y
    grad_w = (2.0 / len(y)) * (X.T @ error)
    grad_b = 2.0 * np.mean(error)
    return grad_w, grad_b

def gradient_step(w, b, grad_w, grad_b, learning_rate):
    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b
    return w_new, b_new

def train_linear(X_train, y_train, learning_rate, steps=1500):
    w, b = np.zeros(X_train.shape[1]), 0.0
    losses = []
    for step in range(steps):
        prediction = linear_predict(X_train, w, b)
        loss = float(mse_loss(y_train, prediction))
        if not np.isfinite(loss) or loss > 1e15:
            return {"w": w, "b": b, "loss": losses, "diverged": True}
        losses.append(loss)
        gw, gb = linear_gradients(X_train, y_train, w, b)
        w, b = gradient_step(w, b, gw, gb, learning_rate)
    return {"w": w, "b": b, "loss": losses, "diverged": False}
"""

# Cell 18: Written answers with Q1 drafted
q1_answer = (
    "The dataset uses a chronological split (Training: Sept 1–Oct 31, 11,520 model-ready rows; "
    "Validation: Nov 1–15, 2,880 rows; Test: Nov 16–30, 2,880 rows) because rental demand is an operational "
    "time-series process where forecasts must rely exclusively on historical observations. Shuffling or random splitting "
    "would violate temporal causality by allowing past predictions to train on future observations. "
    "Standardization must compute statistics (mean and standard deviation) on training rows only (where maximum "
    "absolute training feature mean is ~3.25e-15 after centering) so that future distribution parameters, such as "
    "seasonal demand shifts or weather events in November, do not leak into feature representations during training. "
    "Finally, including the supplied column `high_demand_threshold` (or retrospective flags like `target_zero_assumed`) "
    "as a feature would cause direct target leakage because `high_demand_threshold` explicitly defines the station-specific "
    "75th percentile cutoff used to assign the ground-truth `target_high_demand` label."
)

cell_18_text = nb["cells"][18]["source"]
new_cell_18_lines = []
for i, line in enumerate(cell_18_text):
    if line.strip().startswith("**Answer:**") and i > 0 and "Q1" in cell_18_text[i - 2]:
        new_cell_18_lines.append(f"**Answer:** {q1_answer}\n")
    else:
        new_cell_18_lines.append(line)

nb["cells"][6]["source"] = [line + "\n" for line in cell_6_code.split("\n")[:-1]]
nb["cells"][8]["source"] = [line + "\n" for line in cell_8_code.split("\n")[:-1]]
nb["cells"][18]["source"] = new_cell_18_lines

with open("Group_01_YouBike.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

with open("YouBike_Starter_EN.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully saved Group_01_YouBike.ipynb and updated YouBike_Starter_EN.ipynb")
