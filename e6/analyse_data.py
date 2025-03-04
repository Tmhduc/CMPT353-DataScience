import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

data_file = "data.csv"
data_df = pd.read_csv(data_file)

mean_times = data_df.groupby("algorithm").mean()["time_taken"].reset_index()
mean_times_sorted = mean_times.sort_values(by="time_taken", ascending=False)

# Rank algorithms by speed (lower time is better rank)
mean_times_sorted["rank"] = mean_times_sorted["time_taken"].rank(method="dense")

algorithms = data_df["algorithm"].unique()
new_dict = {}
for alg in algorithms:
    time_groups = data_df[data_df["algorithm"] == alg]["time_taken"]
    # print(time_groups.dtypes)
    # print(time_groups.values)
    new_dict.update({alg: time_groups.values})

new_data_df = pd.DataFrame(new_dict)
new_data_melt = pd.melt(new_data_df)
values_list = list(new_dict.values())
f_stat, p_value = stats.f_oneway(values_list[0], values_list[1], values_list[2], values_list[3], values_list[4], values_list[5], values_list[6])    
turkey_results = pairwise_tukeyhsd(new_data_melt["value"], new_data_melt["variable"], alpha=0.5)
fig = turkey_results.plot_simultaneous()
plt.savefig(fig)
# Print results
print("Sorting Implementations and Their Mean Execution Time:")
print(mean_times_sorted[["algorithm", "time_taken"]])
print("\nSorting Implementations Ranked by Speed (Lower is Faster):")
print(mean_times_sorted[["algorithm", "rank"]])
print("\nANOVA Results:")
print(f"F-statistic = {f_stat:.3f}, p-value = {p_value:.8f}")
print("\nPairwise Tukey's HSD Test Results:")
print(turkey_results)
