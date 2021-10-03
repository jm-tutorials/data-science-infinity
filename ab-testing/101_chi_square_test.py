##################################################
# AB Testing - comparing to mailers ( Chi Squared)
##################################################

import pandas as pd
from scipy.stats import chi2_contingency, chi2

campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name="campaign_data") 
# Filter our data
campaign_data2 = campaign_data.loc[campaign_data.mailer_type != "Control"]

# Summarise to get our observed frequencies
observed_values = pd.crosstab(campaign_data2.mailer_type, campaign_data2.signup_flag).values
mailer1_signup_rate = observed_values[0][1]/sum(observed_values[0])
mailer2_signup_rate = observed_values[1][1]/sum(observed_values[1])

# state hypotheses and set acceptance criteria
null_hypothesis = "There is no relationship between mailer type and signup rate. They are independent"
alternative_hypothesis = "There is a relationship between mailer and sinup rate. They are not independent"
acceptance_criteria = 0.05

# calculate expected frequencies and chie squared statistic
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)

# find the critical value for our test
critical_value = chi2.ppf(1 - acceptance_criteria, dof)
if chi2_statistic >= critical_value:
    print(f"As our chi-squared statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our chi-squared statistic of {chi2_statistic} is less than our critical value of {critical_value} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}")

# print the results (p-value)
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is higher than our acceptance_criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}")
else:
    print(f"As our p-value of {p_value} is less than our acceptance_criteria of {acceptance_criteria} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}")
