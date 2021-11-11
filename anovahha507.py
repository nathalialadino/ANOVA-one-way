# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 01:37:45 2021

@author: natha
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# dataframe transformationg and altering
df2 = pd.read_csv('C:/Users/natha/OneDrive/Documents/deathcountscsv2.csv')
data2 = df2.rename(columns={"Age Group": "age", "Race or Ethnicity": "race", "Selected Cause of Death": "cause"})

recenty = data2[data2.Year != 2015]
recent2 = recenty[data2.Year != 2016]
recent3 = recent2[data2.Year != 2017]
recent4 = recent3[data2.Year != 2003]
recent5 = recent4[data2.Year != 2014]
recent6 = recent5[data2.Year != 2004]
recent7 = recent6[data2.Year != 2012]
recent8 = recent7[data2.Year != 2013]
recent9 = recent8[data2.Year != 2005]
recent10 = recent9[data2.Year != 2011]
recent11 = recent10[data2.Year != 2010]
recent12 = recent11[data2.Year != 2006]
recent13 = recent12[data2.Year != 2009]
recent14 = recent13[data2.Year != 2008]
data2018 = recent14[data2.Year != 2007]

datatotal1 = data2018[data2.cause != 'All Other Causes']
datatotal2 = datatotal1[data2.cause != 'Diseases of the Heart']
datatotal3 = datatotal2[data2.cause != 'Malignant Neoplasms']
datatotal4 = datatotal3[data2.cause != 'Accidents']
datatotal5 = datatotal4[data2.cause != 'Cerebrovascular Disease']
datatotal6 = datatotal5[data2.cause != 'CLRD']
datatotal7 = datatotal6[data2.cause != 'Pneumonia']
datatotal8 = datatotal7[data2.cause != 'Diabetes Mellitus']
Datatotal = datatotal8[data2.cause != 'AIDS']

### IV1 - Age group
aggregation_functions = {'Deaths': 'sum', 'age': 'first'}
df_new2 = Datatotal.groupby(Datatotal['age']).aggregate(aggregation_functions)

df_new2.index.names = ['Index']
df_new2.reset_index(drop=True, inplace=True)
df_new2.reset_index()
print(df_new2)

import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf 

# Testing normality using the Shapiro test

modela = smf.ols("Deaths ~ C(age)", data = df_new2).fit()
stats.shapiro(modela.resid) 

# ShapiroResult(statistic=0.9578123688697815, pvalue=0.7441322803497314)
# ShapiroResult(statistic=0.9199321269989014, pvalue=0.3180749714374542)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111) 

normality_plot, stat = stats.probplot(modela.resid, plot= plt, rvalue=True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show() 


# grouping of age groups to beter reach equal variances and levene test

young = [227, 507, 802, 956]
middle = [2693, 3637, 8244, 18639]
old = [27787, 36870, 56750]


import numpy as np
Young = np.array(young, int)
print(type(Young[0]))

import numpy as np
Middle = np.array(middle, int)
print(type(Middle[0]))

import numpy as np
Old = np.array(old, int)
print(type(Old[0]))

plt.hist(Young)
plt.show()

plt.hist(Middle)
plt.show()

plt.hist(Old)
plt.show()

from scipy.stats import kurtosis, skew
import numpy as np


print(kurtosis(Young))
print(kurtosis(Middle))
print(kurtosis(Old)) 

print(skew(Young))
print(skew(Middle))
print(skew(Old)) 


import scipy.stats as stats


## Levene test for assumption of variance


stats.levene(Young, Middle, Old)
# LeveneResult(statistic=2.2298672099733308, pvalue=0.16995142341791253)
# other way

young = [227, 507, 802, 956]
middle = [2693, 3637, 8244, 18639]
old = [27787, 36870, 56750]

alpha = 0.05

w,pvalue = stats.levene(young, middle, old)
print(w, pvalue)

if pvalue > alpha :
  print("We do not reject the null hypothesis")
else:
  print("Reject the Null Hypothesis")
  

# 2.2298672099733308 0.16995142341791253
# We do not reject the null hypothesis

## One way ANOVA for levels of Age group and Total Death Counts in 2018
# Question: Is there a difference between the different levels of age
# group and total death count?

stats.f_oneway(Young, Middle, Old)

# F_onewayResult(statistic=19.668126721251667, pvalue=0.0008158013541395036)

## post hoc test
import statsmodels.stats.multicomp as mc
comp = mc.MultiComparison(Datatotal['Deaths'], Datatotal['age'])
post_hoc_res = comp.tukeyhsd()
tukey1way = pd.DataFrame(post_hoc_res.summary())

# Interpretation: The difference between the two variables is significant.

### IV2 - Cause of death


### Cause of death compared with death counts


datac = data2018[data2018.cause != 'Total']
datac['cause'].value_counts()

aggregation_functions2 = {'cause': 'first', 'Deaths': 'sum'}
datanewc = datac.groupby(datac['cause']).aggregate(aggregation_functions2)

datanewc.index.names = ['Index']
datanewc.reset_index(drop=True, inplace=True)
datanewc.reset_index()
print(datanewc)



modelc = smf.ols("Deaths ~ C(cause)", data = datanewc).fit()
stats.shapiro(modelc.resid)

# ShapiroResult(statistic=0.9742709398269653, pvalue=0.9285860061645508)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111) 

normality_plot, stat = stats.probplot(modelc.resid, plot= plt, rvalue=True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show() 

cause1 = datac[datac['cause'] == 'All Other Causes']
cause2 = datac[datac['cause'] == 'Diseases of the Heart']
cause3 = datac[datac['cause'] == 'Malignant Neoplasms']
cause4 = datac[datac['cause'] == 'Accidents']
cause5 = datac[datac['cause'] == 'Cerebrovascular Disease']
cause6 = datac[datac['cause'] == 'CLRD']
cause7 = datac[datac['cause'] == 'Pneumonia']
cause8 = datac[datac['cause'] == 'Diabetes Mellitus']
cause9 = datac[datac['cause'] == 'AIDS']


plt.hist(cause1['Deaths'])
plt.show()

plt.hist(cause2['Deaths'])
plt.show()

plt.hist(cause3['Deaths'])
plt.show()

plt.hist(cause4['Deaths'])
plt.show()

plt.hist(cause5['Deaths'])
plt.show()

plt.hist(cause6['Deaths'])
plt.show()

plt.hist(cause7['Deaths'])
plt.show()

plt.hist(cause8['Deaths'])
plt.show()

plt.hist(cause9['Deaths'])
plt.show()

stats.levene(cause1['Deaths'],
               cause2['Deaths'],
               cause3['Deaths'],
               cause4['Deaths'],
               cause5['Deaths'],
               cause6['Deaths'],
               cause7['Deaths'],
               cause8['Deaths'],
               cause9['Deaths']
               )


from scipy import stats

# Question: Is there a difference between the different reasons of cause of 
# death and the death counts?

stats.f_oneway(cause1['Deaths'],
               cause2['Deaths'],
               cause3['Deaths'],
               cause4['Deaths'],
               cause5['Deaths'],
               cause6['Deaths'],
               cause7['Deaths'],
               cause8['Deaths'],
               cause9['Deaths']
               )  

# F_onewayResult(statistic=19.668126721251667, pvalue=0.0008158013541395036)

stats.kruskal(cause1['Deaths'],
               cause2['Deaths'],
               cause3['Deaths'],
               cause4['Deaths'],
               cause5['Deaths'],
               cause6['Deaths'],
               cause7['Deaths'],
               cause8['Deaths'],
               cause9['Deaths']
               )  
# KruskalResult(statistic=150.52208013304755, pvalue=1.5259083337184075e-28)
# Interpretation: The difference between the two variables is significant.



### IV3 - Race

aggregation_functions3 = {'race': 'first', 'Deaths': 'sum'}
datanewr = Datatotal.groupby(Datatotal['race']).aggregate(aggregation_functions3)

datanewr.index.names = ['Index']
datanewr.reset_index(drop=True, inplace=True)
datanewr.reset_index()
print(datanewr)


modelr = smf.ols("Deaths ~ C(race)", data = datanewr).fit()
stats.shapiro(modelr.resid)

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111) 

normality_plot, stat = stats.probplot(modelr.resid, plot= plt, rvalue=True)
ax.set_title("Probability plot of regression residuals \n with R value")
ax.set
plt.show() 

race1 = Datatotal[Datatotal['race'] == 'Black Non Hispanic']
race2 = Datatotal[Datatotal['race'] == 'Hispanic']
race3 = Datatotal[Datatotal['race'] == 'Not Stated']
race4 = Datatotal[Datatotal['race'] == 'Other Non Hispanic']
race5 = Datatotal[Datatotal['race'] == 'White Non Hispanic']

stats.levene(race1['Deaths'],
               race2['Deaths'],
               race3['Deaths'],
               race4['Deaths'],
               race5['Deaths']
               )

plt.hist(race1['Deaths'])
plt.show()

plt.hist(race2['Deaths'])
plt.show()

plt.hist(race3['Deaths'])
plt.show()

plt.hist(race4['Deaths'])
plt.show()

plt.hist(race5['Deaths'])
plt.show()

# Question: Is there a difference between the different levels of race and 
# the total death counts?

stats.f_oneway(race1['Deaths'],
               race2['Deaths'],
               race3['Deaths'],
               race4['Deaths'],
               race5['Deaths']
               )
# F_onewayResult(statistic=8.34126682891993, pvalue=6.982189166298508e-06)

stats.kruskal(race1['Deaths'],
               race2['Deaths'],
               race3['Deaths'],
               race4['Deaths'],
               race5['Deaths']
               )
# KruskalResult(statistic=29.286285926477237, pvalue=6.837350251052797e-06)

# Interpretation: The difference between the two variables is significant.
