##############################
# Independent Samples T-Test
##############################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, norm

sample_a = norm.rvs(loc=500, scale=100, size=250, random_state=42).astype(int)
sample_b = norm.rvs(loc=550, scale=150, size=100, random_state=42).astype(int)

plt.hist(sample_a, density=True, alpha=0.5)
plt.hist(sample_b, density=True, alpha=0.5)
plt.show()
sample_a_mean = sample_a.mean()
sample_b_mean = sample_b.mean()

null_hypothesis = "The mean of the sample A is equal to the mean of sample B"
alternative_hypothesis = "The mean of the sample A is different to the mean of sample B"
acceptance_criteria = 0.05

t_statistic, p_value = ttest_ind(sample_a, sample_b)

# print the results (p-value)
print("Independent Sample T-Test:")
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our p-value of {p_value} is less than our acceptance_criteria of {acceptance_criteria} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}")

##############################
# Welch's T-Test
##############################

t_statistic, p_value = ttest_ind(sample_a, sample_b, equal_var=False)

print("\nWelch's T-Test:")
# print the results (p-value)
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our p-value of {p_value} is less than our acceptance_criteria of {acceptance_criteria} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}")
