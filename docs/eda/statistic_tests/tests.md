## **One sample t-test**
The One Sample t Test determines whether the sample mean is statistically different from a known or hypothesised population mean. The One Sample t Test is a parametric test.

**Example**: you have 10 ages and you are checking whether avg age is 30 or not.
```python
from scipy.stats import ttest_1samp
import numpy as np

ages = np.genfromtxt(“ages.csv”)

print(ages)

ages_mean = np.mean(ages)
print(ages_mean)

tset, pval = ttest_1samp(ages, 30)
print(“p-values”,pval)

if pval < 0.05:
    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")
```

## **Two sampled T-test**
The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different. The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test

```python
from scipy.stats

import ttest_ind
import numpy as np

week1 = np.genfromtxt("week1.csv",  delimiter=",")
week2 = np.genfromtxt("week2.csv",  delimiter=",")

week1_mean = np.mean(week1)
week2_mean = np.mean(week2)

print("week1 mean value:",week1_mean)
print("week2 mean value:",week2_mean)

week1_std = np.std(week1)
week2_std = np.std(week2)

print("week1 std value:",week1_std)
print("week2 std value:",week2_std)

ttest,pval = ttest_ind(week1,week2)
print("p-value",pval)

if pval <0.05:
    print("we reject null hypothesis")
else:
    print("we accept null hypothesis")
```

## **Paired sampled t-test**
The paired sample t-test is also called dependent sample t-test. It’s an uni variate test that tests for a significant difference between 2 related variables. An example of this is if you where to collect the blood pressure for an individual before and after some treatment, condition, or time point.

**H0:** means difference between two sample is 0

**H1:** mean difference between two sample is not 0

```python
import pandas as pd
from scipy import stats

df = pd.read_csv("blood_pressure.csv")

df[['bp_before','bp_after']].describe()

ttest,pval = stats.ttest_rel(df['bp_before'], df['bp_after'])
print(pval)

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
```

## **When you can run a Z Test.**

Several different types of tests are used in statistics (i.e. f test, chi square test, t test). You would use a Z test if:

- Your sample size is greater than 30. Otherwise, use a t test.

- Data points should be independent from each other. In other words, one data point isn’t related or doesn’t affect another data point.

- Your data should be normally distributed. However, for large sample sizes (over 30) this doesn’t always matter.

- Your data should be randomly selected from a population, where each item has an equal chance of being selected.
Sample sizes should be equal if at all possible.

```python
import pandas as pd
from scipy import stats
from statsmodels.stats
import weightstats as stests

ztest, pval = stests.ztest(df['bp_before'], x2=None, value=156)

print(float(pval))

if pval<0.05:
    print("reject null hypothesis")
else:
    print("not reject null hypothesis")
```

## **Two-sample Z test**
In two sample z-test, similar to t-test here we are checking two independent data groups and deciding whether sample mean of two group is equal or not.

**H0: mean of two group is 0**

**H1: mean of two group is not 0**

```python
ztest, pval1 = stests.ztest(df['bp_before'],
                            x2=df['bp_after'],
                            value=0,
                            alternative='two-sided')

print(float(pval1))

if pval<0.05:
    print("reject null hypothesis")
else:
    print("not reject null hypothesis")
```

## **ANOVA (F-TEST)**
The t-test works well when dealing with two groups, but sometimes we want to compare more than two groups at the same time. For example, if we wanted to test whether voter age differs based on some categorical variable like race, we have to compare the means of each level or group the variable. We could carry out a separate t-test for each pair of groups, but when you conduct many tests you increase the chances of false positives. The analysis of variance or ANOVA is a statistical inference test that lets you compare multiple groups at the same time.

**F = Between group variability / Within group variability**

Unlike the z and t-distributions, the F-distribution does not have any negative values because between and within-group variability are always positive due to squaring each deviation.

## **One Way F-test(Anova)**
It tell whether two or more groups are similar or not based on their mean similarity and f-score.

Example : there are 3 different category of plant and their weight and need to check whether all 3 group are similar or not

```python
df_anova = pd.read_csv('PlantGrowth.csv')
df_anova = df_anova[['weight','group']]
grps = pd.unique(df_anova.group.values)
d_data = {grp:df_anova['weight'][df_anova.group == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])

print("p-value for significance is: ", p)

if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
```

## **Two Way F-test**
Two way F-test is extension of 1-way f-test, it is used when we have 2 independent variable and 2+ groups. 2-way F-test does not tell which variable is dominant. if we need to check individual significance then Post-hoc testing need to be performed.

Now let’s take a look at the Grand mean crop yield (the mean crop yield not by any sub-group), as well the mean crop yield by each factor, as well as by the factors grouped together

```python
import statsmodels.api as sm
from statsmodels.formula.api 
import ols

df_anova2 = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/crop_yield.csv")

model = ols('Yield ~ C(Fert)*C(Water)', df_anova2).fit()

print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

res = sm.stats.anova_lm(model, typ= 2)
res
```

## **Chi-Square Test**
The test is applied when you have two categorical variables from a single population. It is used to determine whether there is a significant association between the two variables.

For example, in an election survey, voters might be classified by gender (male or female) and voting preference (Democrat, Republican, or Independent). We could use a chi-square test for independence to determine whether gender is related to voting preference

```python
import pandas as pd
from scipy.stats
import chi2

df_chi = pd.read_csv('chi-test.csv')
contingency_table=pd.crosstab(df_chi["Gender"],df_chi["Shopping?"])

print('contingency_table :-\n',contingency_table)#Observed Values

Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)

b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)

print("Degree of Freedom:-",ddof)
alpha = 0.05

chi_square = sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

chi_square_statistic = chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value: ',critical_value)#p-value

p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

print('p-value: ', p_value)
print('Significance level: ', alpha)
print('Degree of Freedom: ', ddof)
print('chi-square statistic: ', chi_square_statistic)
print('critical_value: ', critical_value)
print('p-value: ', p_value)

if chi_square_statistic >= critical_value:
    print("Reject H0, There is a relationship between 2 categorical variables")
else:
    print("Retain H0, There is no relationship between 2 categorical variables")
    
if p_value <= alpha:
    print("Reject H0, There is a relationship between 2 categorical variables")
else:
    print("Retain H0, There is no relationship between 2 categorical variables")
```
