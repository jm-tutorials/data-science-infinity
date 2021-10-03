## business hypothesis:
# Because we have seem good success through in-app notifications in the past, if
# send an in-app notification with a promotional offer for Lamps, then the % of 
# users that purchase from the Lamp Category will increase
# Primary metric - Transaction Rate i.e % of users that will make a purchase
# Secondary metrics - Purchase Value
# Other metrics - Uninstall rate

# install.packages("pwr")
library(pwr)
# install.packages("glue")
library(glue)

setwd("~/Documents/learning/data-science-infinity/ab-testing")

control <- 0.101
uplift <- .2 # we want at least 20% uplfit
variant <- (1 + uplift) * control
effect_size <- ES.h(control, variant)
sample_size_output <- pwr.2p.test(h = effect_size,
                                 n = ,
                                 sig.level = 0.05,
                                 power = 0.8)

sample_size_output <- ceiling(sample_size_output$n)
glue('sample size needed: {sample_size_output}')


############################## Understanding the data
df <- read.csv("udemy_ab_test_data.csv")

# Treatment indicator - allocation
# Response variables - addtocart_flag, transaction_flag, purchase_value
# Baseline variables - active_6m, days_since
# Other - uninstall flag

# Numerical variable stats
summary(df[,c("active_6m", "addtocart_flag", "transaction_flag", "uninstall_flag",
              "purchase_value", "days_since")])

# install.packages(c("ggplot2","dplyr"))
library(ggplot2)
library(dplyr)

# plot purchase value
df %>% 
  ggplot(aes(x = purchase_value)) +
  geom_histogram( color="#e9ecef", fill="#E69F00", alpha=0.6, position='identity') +
  scale_fill_manual(values=c("#69b3a2", "404080")) +
  labs(fill="")

df %>% ggplot(aes(x=purchase_value)) +
  geom_density(color="#E69F00")

# plot days_since
df %>% ggplot(aes(x = days_since)) +
  geom_histogram( color="#e9ecef", fill="#56B4E9", alpha=0.6, position='identity') +
  scale_fill_manual(values=c("#69b3a2", "404080")) +
  labs(fill="")

df %>% ggplot(aes(x=days_since)) +
  geom_density(color="#56B4E9")

# categorical variable stats
table(df$allocation)
table(df$allocation)/nrow(df)


# ############################## Check for randomization - baseline variables
df %>% 
  group_by(allocation) %>%
  summarise(mean(active_6m), mean(days_since))

# compare distribution across the two groups

df %>% 
  ggplot(aes(x=days_since, fill=allocation)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position='identity') +
  scale_fill_manual(values=c("#69b3a2", "404080")) +
  labs(fill="")

############################## Treatment Effects
# Compare performance of response variables across the two groups
df  %>%
  group_by(allocation) %>%
  summarise(mean(addtocart_flag),
            mean(transaction_flag),
            mean(purchase_value, na.rm=TRUE))

prop.test(xtabs(~ allocation + transaction_flag, data=df)[,2:1])

# This is tellin
  

