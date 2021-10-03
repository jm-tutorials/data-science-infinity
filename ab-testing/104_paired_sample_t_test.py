##############################
# Paired Samples T-Test
##############################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, norm

before = norm.rvs(loc=500, scale=100, size=100, random_state=42).astype(int)

np.random.seed(42)
after = before + np.random.randint(low=-50, high=75, size=100)

plt.hist(before, density=True, alpha=0.5, label='before')
plt.hist(after, density=True, alpha=0.5, label='after')
plt.legend()
plt.show()
before_mean = before.mean()
after_mean = after.mean()

null_hypothesis = "The mean of the before sample is equal to the mean of after sample"
alternative_hypothesis = "The mean of the before sample is different to the mean of after sample"
acceptance_criteria = 0.05

t_statistic, p_value = ttest_rel(before, after)

# print the results (p-value)
print("Independent Sample T-Test:")
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is lower than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}")