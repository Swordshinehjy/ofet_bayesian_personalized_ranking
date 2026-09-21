import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def leading_digit(x):
    x = abs(x)
    if x == 0:
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)

def benford_probability(d):
    return np.log10(1 + 1 / d)

df = pd.read_csv(r"d:\code\ml-ofet\ofet_bayesian_personalized_ranking\contrastive_paired.csv")

mobility_cols = ["mu_e_1", "mu_h_1", "mu_e_2", "mu_h_2"]
all_values = pd.concat([df[c] for c in mobility_cols]).dropna()
all_values = all_values[all_values > 0]

digits = all_values.apply(leading_digit).dropna().astype(int)

observed = digits.value_counts().sort_index()
observed = observed.reindex(range(1, 10), fill_value=0)

total = observed.sum()
expected_pct = pd.Series({d: benford_probability(d) for d in range(1, 10)})
expected = (expected_pct * total).values
observed_vals = observed.values

chi2_stat, p_value = chisquare(observed_vals, f_exp=expected)

print("=" * 60)
print("Benford's Law Test Results")
print("=" * 60)
print(f"\nValid mobility data count: {total}")
print(f"Data source columns: {mobility_cols}")
print(f"\n{'Digit':<10}{'Observed':<12}{'Benford Exp':<12}{'Diff':<10}")
print("-" * 44)
for d in range(1, 10):
    obs_freq = observed_vals[d - 1] / total
    exp_freq = expected_pct[d]
    diff = obs_freq - exp_freq
    print(f"{d:<10}{obs_freq:<12.4f}{exp_freq:<12.4f}{diff:<+10.4f}")

print(f"\nChi-squared test:")
print(f"  χ² statistic = {chi2_stat:.4f}")
print(f"  p-value      = {p_value:.6f}")
if p_value > 0.05:
    print(f"  Conclusion: p={p_value:.4f} > 0.05, data follows Benford's Law ✓")
else:
    print(f"  Conclusion: p={p_value:.4f} ≤ 0.05, data does NOT follow Benford's Law ✗")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax1 = axes[0]
x = np.arange(1, 10)
width = 0.35
obs_freq = observed_vals / total
exp_freq = expected_pct.values
ax1.bar(x - width / 2, obs_freq, width, label="Observed", color="#4C72B0", alpha=0.85)
ax1.bar(x + width / 2, exp_freq, width, label="Benford Expected", color="#DD8452", alpha=0.85)
ax1.set_xlabel("Leading Digit", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title(f"Benford's Law Test (χ²={chi2_stat:.2f}, p={p_value:.4f})", fontsize=13)
ax1.set_xticks(x)
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.3)

ax2 = axes[1]
log_values = np.log10(all_values)
ax2.hist(log_values, bins=30, color="#55A868", alpha=0.8, edgecolor="white")
ax2.set_xlabel("log₁₀(Mobility)", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Distribution of log₁₀(Mobility)", fontsize=13)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(r"d:\code\ml-ofet\ofet_bayesian_personalized_ranking\benford_test.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nChart saved to benford_test.png")
