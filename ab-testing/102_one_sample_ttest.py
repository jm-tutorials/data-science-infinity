##############################
# One Sample T-Test
##############################
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, norm

population = norm.rvs(loc=500, scale=100, size=1000, random_state=42).astype(int)

np.random.seed(42)
sample = np.random.choice(population, 250)

plt.hist(population, density=True, alpha=0.5)
plt.hist(sample, density=True, alpha=0.5)
plt.show()

population_mean = population.mean()
sample_mean = sample.mean()

null_hypothesis = "The mean of the sample is equal to the mean of the population"
alternative_hypothesis = "The mean of the sample is different to the mean of the population"
acceptance_criteria = 0.05

t_statistic, p_value = ttest_1samp(sample, population_mean)
print(t_statistic, p_value)

# print the results (p-value)
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our p-value of {p_value} is less than our acceptance_criteria of {acceptance_criteria} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}")

