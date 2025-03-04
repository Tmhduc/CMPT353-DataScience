import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats, levene

# Generate sample data
np.random.seed(42)
group1 = np.random.normal(loc=75, scale=10, size=30)  # Mean=75, Std=10
group2 = np.random.normal(loc=80, scale=30, size=30)  # Mean=80, Std=30 (higher variance)

# Perform Levene’s test to check variance equality
stat, p_val_levene = levene(group1, group2)
print(f"Levene’s Test p-value: {p_val_levene:.3f}")

# Standard t-test (assumes equal variance)
t_stat, p_val_ttest = ttest_ind(group1, group2, equal_var=True)
print(f"Standard t-test p-value: {p_val_ttest:.3f}")

# Welch’s t-test (corrects for unequal variance)
t_stat_welch, p_val_welch = ttest_ind(group1, group2, equal_var=False)
print(f"Welch’s t-test p-value: {p_val_welch:.3f}")

# Interpretation
if p_val_ttest < 0.05:
    print("Standard t-test: Incorrectly rejects H0 due to variance issues.")
else:
    print("Standard t-test: Fails to detect true difference.")

if p_val_welch < 0.05:
    print("Welch’s t-test: Correctly detects the true difference.")
else:
    print("Welch’s t-test: No significant difference detected.")
