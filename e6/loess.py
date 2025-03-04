import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Generate noisy data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = np.sin(x)  # Underlying true function
y_noisy = y_true + np.random.normal(scale=0.3, size=len(x))  # Adding noise

# Apply LOESS smoothing
smoothed = lowess(y_noisy, x, frac=0.2, it=3, return_sorted=False)  # frac=bandwidth

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(x, y_noisy, color='gray', alpha=0.5, label="Noisy Data")
plt.plot(x, y_true, linestyle="dashed", color='blue', label="True Function")
plt.plot(x, smoothed, color='red', linewidth=2, label="LOESS Smoothed Curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("LOESS Smoothing Example")
plt.show()
