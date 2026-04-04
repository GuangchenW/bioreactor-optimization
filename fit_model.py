import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# --- Load data --- #
df = pd.read_csv("data/base.csv")
x_data = df["input"].values
y_data = ((df["Flat"] + df["50x50"] + df["15x15"]) / 3).values


# --- Model definitions --- #
def model_parsimonious(x, a, b):
    """f(x) = a*x^4 / (b + x^5) — 2-param integer-exponent Hill form."""
    return a * x**4 / (b + x**5)


def model_general(x, a, p, b, q):
    """f(x) = a*x^p / (b + x^q) — 4-param generalized Hill form."""
    return a * x**p / (b + x**q)


# --- Fit metrics --- #
def compute_metrics(y_obs, y_pred, k):
    n = len(y_obs)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / n)
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + k * np.log(n)
    return {"R2": r2, "RMSE": rmse, "AIC": aic, "BIC": bic}


# --- Fit models --- #
popt_p, pcov_p = curve_fit(
    model_parsimonious, x_data, y_data, p0=[0.13, 0.06], maxfev=50000
)
popt_g, pcov_g = curve_fit(
    model_general, x_data, y_data, p0=[0.12, 4.4, 0.034, 5.3], maxfev=50000
)

perr_p = np.sqrt(np.diag(pcov_p))
perr_g = np.sqrt(np.diag(pcov_g))

y_pred_p = model_parsimonious(x_data, *popt_p)
y_pred_g = model_general(x_data, *popt_g)

metrics_p = compute_metrics(y_data, y_pred_p, k=2)
metrics_g = compute_metrics(y_data, y_pred_g, k=4)


# --- Print results --- #
print("=" * 60)
print("MODEL 1: f(x) = ax^4 / (b + x^5)")
print(f"  a = {popt_p[0]:.6f} ± {perr_p[0]:.6f}")
print(f"  b = {popt_p[1]:.6f} ± {perr_p[1]:.6f}")
print(f"  {metrics_p}")
peak_x = (4 * popt_p[1]) ** 0.2
print(f"  Peak at x* = (4b)^(1/5) = {peak_x:.4f}")
print(f"  Peak value = {model_parsimonious(peak_x, *popt_p):.6f}")
print()
print("MODEL 2: f(x) = ax^p / (b + x^q)")
print(f"  a = {popt_g[0]:.6f} ± {perr_g[0]:.6f}")
print(f"  p = {popt_g[1]:.6f} ± {perr_g[1]:.6f}")
print(f"  b = {popt_g[2]:.6f} ± {perr_g[2]:.6f}")
print(f"  q = {popt_g[3]:.6f} ± {perr_g[3]:.6f}")
print(f"  {metrics_g}")
print("=" * 60)


# --- Plot --- #
x_fine = np.linspace(0.01, 11, 500)

fig, axes = plt.subplots(
    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
)

ax = axes[0]
ax.scatter(x_data, y_data, c="black", s=60, zorder=5, label="Observations")
ax.plot(
    x_fine,
    model_parsimonious(x_fine, *popt_p),
    "b-",
    lw=2,
    label=(
        f"$f(x) = ax^4/(b+x^5)$\n"
        f"  a={popt_p[0]:.4f}, b={popt_p[1]:.4f}, $R^2$={metrics_p['R2']:.4f}"
    ),
)
ax.plot(
    x_fine,
    model_general(x_fine, *popt_g),
    "r--",
    lw=2,
    alpha=0.8,
    label=(
        f"$f(x) = ax^p/(b+x^q)$\n"
        f"  a={popt_g[0]:.4f}, p={popt_g[1]:.2f}, "
        f"b={popt_g[2]:.4f}, q={popt_g[3]:.2f}, $R^2$={metrics_g['R2']:.4f}"
    ),
)
ax.set_xlabel("Input (O2 flow rate)", fontsize=12)
ax.set_ylabel("Output (O2 concentration delta)", fontsize=12)
ax.set_title("Model Fitting: Generalized Hill Function", fontsize=14)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(
    x_data, y_data - y_pred_p, c="blue", s=40, label=f"$ax^4/(b+x^5)$ residuals"
)
ax2.scatter(
    x_data,
    y_data - y_pred_g,
    c="red",
    s=40,
    marker="x",
    label=f"$ax^p/(b+x^q)$ residuals",
)
ax2.axhline(0, color="black", lw=0.5)
ax2.set_xlabel("Input (O2 flow rate)", fontsize=12)
ax2.set_ylabel("Residual", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model_fit.png", dpi=150, bbox_inches="tight")
plt.show()
